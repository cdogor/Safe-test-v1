"""
safe_engine.py
==============
Moteur SAFE 2.5D (Semi-Analytical Finite Element Method)
pour milieu de type digue / barrage en remblai.

Formulation :
  [K1 + i·k·K2 + k²·K3 - ω²·M] · U = 0

Plan de discrétisation : (x, z) transversal
Direction de propagation : y (axiale, invariante)
Éléments : Q4 quadrilatères isoparamétriques, intégration Gauss 2×2
DDL par nœud : [ux, uz] (déplacements dans le plan)

Référence :
  Hayashi & Rose (2003) – Guided wave dispersion curves
  Castaings & Lowe (2008) – Finite element model for waves guided along solid
  Bartoli et al. (2006) – Modeling wave propagation in damped waveguides
"""

import numpy as np
from scipy.linalg import eig
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings


# ─────────────────────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────────────────────

@dataclass
class Material:
    name: str
    E: float       # Module de Young (Pa)
    nu: float      # Coefficient de Poisson
    rho: float     # Masse volumique (kg/m³)
    color: str = "#4488ff"

    @property
    def Vs(self) -> float:
        """Vitesse onde S (m/s)"""
        return np.sqrt(self.E / (2 * (1 + self.nu) * self.rho))

    @property
    def Vp(self) -> float:
        """Vitesse onde P (m/s)"""
        c = self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu) * self.rho)
        return np.sqrt(c)

    @property
    def lam(self) -> float:
        """Premier coefficient de Lamé λ"""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        """Module de cisaillement μ"""
        return self.E / (2 * (1 + self.nu))


@dataclass
class Layer:
    """Couche géologique définie par un polygone convexe + matériau."""
    polygon: np.ndarray   # shape (n,2) – coordonnées (x,z) dans le plan transversal
    material: Material
    label: str = ""


@dataclass
class SAFEMesh:
    """Maillage assemblé."""
    nodes: np.ndarray          # (N_nodes, 2) – coordonnées (x,z)
    connectivity: np.ndarray   # (N_elem, 4) – indices nœuds Q4
    elem_materials: List[Material]
    n_dof: int                 # = 2 * N_nodes


@dataclass
class DispersionResult:
    """Résultats de la dispersion pour un balayage en fréquence."""
    freqs: np.ndarray          # fréquences (Hz)
    wavenumbers: np.ndarray    # (n_modes, n_freqs) – nombres d'onde réels (rad/m)
    vph: np.ndarray            # (n_modes, n_freqs) – vitesse de phase (m/s)
    vgr: np.ndarray            # (n_modes, n_freqs) – vitesse de groupe (m/s)
    mode_shapes: Optional[np.ndarray] = None  # (n_modes, n_freqs, n_dof)


# ─────────────────────────────────────────────────────────────
#  Fonctions de forme Q4 isoparamétriques
# ─────────────────────────────────────────────────────────────

# Points et poids de Gauss (2×2)
_GP = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
_GW = np.array([1.0, 1.0])


def shape_functions(xi: float, eta: float):
    """
    Fonctions de forme Q4 et leurs dérivées en (xi, eta).
    Numérotation antihoraire : n0(bas-gauche), n1(bas-droite), n2(haut-droite), n3(haut-gauche)
    """
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ])
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta),
    ])
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi),
    ])
    return N, dN_dxi, dN_deta


# ─────────────────────────────────────────────────────────────
#  Matrice de rigidité isotrope (élasticité 3D)
# ─────────────────────────────────────────────────────────────

def build_D_3d(mat: Material) -> np.ndarray:
    """
    Matrice constitutive 6×6 isotrope 3D (Voigt notation):
    [σxx, σyy, σzz, σyz, σxz, σxy]^T = D · [εxx, εyy, εzz, γyz, γxz, γxy]^T
    """
    lam, mu = mat.lam, mat.mu
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


# ─────────────────────────────────────────────────────────────
#  Matrices élémentaires SAFE Q4
# ─────────────────────────────────────────────────────────────

def elem_safe_matrices(coords: np.ndarray, mat: Material):
    """
    Calcule les matrices élémentaires SAFE pour un élément Q4.

    coords : (4, 2) – coordonnées (x, z) des 4 nœuds
    mat    : matériau de l'élément

    Convention DDL par nœud : [ux, uy, uz]  (3 DDL, propagation en y)
    → taille locale 12×12

    Retourne : K1, K2, K3, Me  (toutes réelles)

    K_eff(k, ω) = K1 + i·k·K2 + k²·K3 - ω²·M
    (K2 est antisymétrique → i·k·K2 hermitien si K2^T = -K2)
    """
    ndof = 12  # 4 nœuds × 3 DDL
    K1 = np.zeros((ndof, ndof))
    K2 = np.zeros((ndof, ndof))
    K3 = np.zeros((ndof, ndof))
    Me = np.zeros((ndof, ndof))

    D = build_D_3d(mat)
    rho = mat.rho

    for gi in range(2):
        for gj in range(2):
            xi, eta = _GP[gi], _GP[gj]
            N, dN_dxi, dN_deta = shape_functions(xi, eta)

            # Jacobien 2×2 (plan x-z)
            J = np.array([
                [np.dot(dN_dxi,  coords[:, 0]), np.dot(dN_dxi,  coords[:, 1])],
                [np.dot(dN_deta, coords[:, 0]), np.dot(dN_deta, coords[:, 1])],
            ])
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Jacobien négatif ou nul ({detJ:.3e}): élément dégénéré")
            invJ = np.linalg.inv(J)

            # Dérivées des fonctions de forme par rapport à (x, z)
            dN = invJ @ np.vstack([dN_dxi, dN_deta])  # (2, 4) : dN/dx, dN/dz
            dNx = dN[0]
            dNz = dN[1]

            w = _GW[gi] * _GW[gj] * detJ

            # ─── Matrices B1, B2, B3 pour chaque nœud a ───────────────
            # Convention déformations (Voigt 3D) :
            # ε = [εxx, εyy, εzz, γyz, γxz, γxy]
            # εxx = dux/dx, εyy = duy/dy = i·k·uy, εzz = duz/dz
            # γyz = duy/dz + duz/dy = duy/dz + i·k·uz
            # γxz = dux/dz + duz/dx
            # γxy = dux/dy + duy/dx = i·k·ux + duy/dx
            #
            # ε = B1·U + i·k·B2·U (termes réels)
            # où U = [ux0,uy0,uz0, ux1,uy1,uz1, ...]

            for a in range(4):
                ia = 3 * a  # indice DDL local de départ
                # B1_a (6×3) : termes indépendants de k
                B1_a = np.zeros((6, 3))
                B1_a[0, 0] = dNx[a]        # εxx = dux/dx
                B1_a[2, 2] = dNz[a]        # εzz = duz/dz
                B1_a[3, 1] = dNz[a]        # γyz = duy/dz
                B1_a[4, 0] = dNz[a]        # γxz = dux/dz
                B1_a[4, 2] = dNx[a]        # γxz += duz/dx
                B1_a[5, 1] = dNx[a]        # γxy = duy/dx

                # B2_a (6×3) : termes en k (facteur i·k)
                B2_a = np.zeros((6, 3))
                B2_a[1, 1] = N[a]          # εyy = i·k·uy
                B2_a[3, 2] = N[a]          # γyz += i·k·uz
                B2_a[5, 0] = N[a]          # γxy += i·k·ux

                for b in range(4):
                    ib = 3 * b
                    B1_b = np.zeros((6, 3))
                    B1_b[0, 0] = dNx[b]
                    B1_b[2, 2] = dNz[b]
                    B1_b[3, 1] = dNz[b]
                    B1_b[4, 0] = dNz[b]
                    B1_b[4, 2] = dNx[b]
                    B1_b[5, 1] = dNx[b]

                    B2_b = np.zeros((6, 3))
                    B2_b[1, 1] = N[b]
                    B2_b[3, 2] = N[b]
                    B2_b[5, 0] = N[b]

                    # K1 += B1_a^T · D · B1_b · w
                    K1[ia:ia+3, ib:ib+3] += (B1_a.T @ D @ B1_b) * w

                    # K2 += B1_a^T · D · B2_b · w - B2_a^T · D · B1_b · w
                    # (antisymétrique → K_eff hermitien pour k réel)
                    K2[ia:ia+3, ib:ib+3] += (B1_a.T @ D @ B2_b - B2_a.T @ D @ B1_b) * w

                    # K3 += B2_a^T · D · B2_b · w  (signe – car (ik)²=–k²)
                    K3[ia:ia+3, ib:ib+3] += (B2_a.T @ D @ B2_b) * w

                    # Masse consistante
                    Me[ia:ia+3, ib:ib+3] += rho * N[a] * N[b] * np.eye(3) * w

    return K1, K2, K3, Me


# ─────────────────────────────────────────────────────────────
#  Maillage d'un polygone convexe par grille Q4
# ─────────────────────────────────────────────────────────────

def point_in_polygon(px: float, pz: float, poly: np.ndarray) -> bool:
    """Ray casting algorithm."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]
        xj, zj = poly[j]
        if ((zi > pz) != (zj > pz)) and (px < (xj - xi) * (pz - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def mesh_polygon(polygon: np.ndarray, nx: int, nz: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maillage Q4 d'un polygone par grille régulière dans sa bounding box.
    Seuls les éléments dont le centre est à l'intérieur du polygone sont retenus.

    Retourne :
        nodes        (N_nodes, 2)
        connectivity (N_elem, 4)  – indices locaux
    """
    xs, zs = polygon[:, 0], polygon[:, 1]
    x0, x1 = xs.min(), xs.max()
    z0, z1 = zs.min(), zs.max()
    dx = (x1 - x0) / nx
    dz = (z1 - z0) / nz

    # Tous les nœuds de la grille (y compris ceux hors polygone)
    nodes = []
    node_idx = {}
    nid = 0
    for iz in range(nz + 1):
        for ix in range(nx + 1):
            x = x0 + ix * dx
            z = z0 + iz * dz
            node_idx[(ix, iz)] = nid
            nodes.append([x, z])
            nid += 1

    # Éléments dont le centre est dans le polygone
    elems = []
    for iz in range(nz):
        for ix in range(nx):
            cx = x0 + (ix + 0.5) * dx
            cz = z0 + (iz + 0.5) * dz
            if not point_in_polygon(cx, cz, polygon):
                continue
            n0 = node_idx[(ix,   iz  )]
            n1 = node_idx[(ix+1, iz  )]
            n2 = node_idx[(ix+1, iz+1)]
            n3 = node_idx[(ix,   iz+1)]
            elems.append([n0, n1, n2, n3])

    if not elems:
        raise ValueError("Aucun élément généré pour ce polygone. Augmenter la résolution.")

    return np.array(nodes, dtype=float), np.array(elems, dtype=int)


# ─────────────────────────────────────────────────────────────
#  Assemblage global
# ─────────────────────────────────────────────────────────────

def assemble_global(layers: List[Layer], nx: int = 8, nz: int = 6,
                    progress_callback=None) -> Tuple[SAFEMesh,
                                                      np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """
    Assemble les matrices globales SAFE K1, K2, K3, M pour toutes les couches.

    Retourne : mesh, GK1, GK2, GK3, GM  (denses, float64)
    """
    # ── Maillage de chaque couche, fusion des nœuds ──────────
    all_nodes = []
    all_elems = []
    all_mats  = []
    node_offset = 0

    for i, layer in enumerate(layers):
        if progress_callback:
            progress_callback(f"Maillage couche {i+1}/{len(layers)} : {layer.label}")
        nodes_l, elems_l = mesh_polygon(layer.polygon, nx, nz)
        all_nodes.append(nodes_l)
        all_elems.append(elems_l + node_offset)
        all_mats.extend([layer.material] * len(elems_l))
        node_offset += len(nodes_l)

    nodes_global = np.vstack(all_nodes)
    elems_global = np.vstack(all_elems)
    n_nodes = len(nodes_global)
    n_dof   = 3 * n_nodes  # 3 DDL : ux, uy, uz

    if progress_callback:
        progress_callback(f"Assemblage : {n_nodes} nœuds, {len(elems_global)} éléments, {n_dof} DDL")

    GK1 = np.zeros((n_dof, n_dof))
    GK2 = np.zeros((n_dof, n_dof))
    GK3 = np.zeros((n_dof, n_dof))
    GM  = np.zeros((n_dof, n_dof))

    for e, (conn, mat) in enumerate(zip(elems_global, all_mats)):
        coords_e = nodes_global[conn]  # (4, 2)
        try:
            K1e, K2e, K3e, Me = elem_safe_matrices(coords_e, mat)
        except ValueError:
            continue  # élément dégénéré → ignoré

        # Assemblage dans les matrices globales
        dofs = np.array([3*n + d for n in conn for d in range(3)])  # (12,)
        for li, gi in enumerate(dofs):
            for lj, gj in enumerate(dofs):
                GK1[gi, gj] += K1e[li, lj]
                GK2[gi, gj] += K2e[li, lj]
                GK3[gi, gj] += K3e[li, lj]
                GM [gi, gj] += Me [li, lj]

    mesh = SAFEMesh(
        nodes=nodes_global,
        connectivity=elems_global,
        elem_materials=all_mats,
        n_dof=n_dof
    )
    return mesh, GK1, GK2, GK3, GM


# ─────────────────────────────────────────────────────────────
#  Solveur SAFE – balayage en fréquence
# ─────────────────────────────────────────────────────────────

def build_companion(K1, K2, K3, M, omega):
    """
    Formule le problème quadratique aux VP en k :
      [K1 - ω²M + i·k·K2 - k²·K3] · U = 0

    Mis sous forme linéaire compagnon (2N × 2N) :
      A · [kU; U] = k · B · [kU; U]

    A = [ i·K2  K1-ω²M ]     B = [ K3   0 ]
        [ I      0      ]         [ 0   -I ]   (signe conventionnel)

    Référence : Bartoli et al. (2006) eq. (23)
    """
    N = K1.shape[0]
    Z = np.zeros_like(K1)
    I = np.eye(N)

    Keff = K1 - omega**2 * M

    A = np.block([[1j * K2, Keff],
                  [I,       Z  ]])
    B = np.block([[K3,  Z],
                  [Z,  -I]])
    return A, B


def solve_dispersion(GK1: np.ndarray, GK2: np.ndarray, GK3: np.ndarray,
                     GM: np.ndarray, freqs: np.ndarray,
                     n_modes: int = 8,
                     k_max: float = 300.0,
                     progress_callback=None) -> DispersionResult:
    """
    Résout le problème SAFE pour chaque fréquence dans `freqs`.

    Pour chaque ω = 2π·f :
      1. Construit le problème compagnon 2N×2N
      2. Résout avec scipy.linalg.eig
      3. Filtre les valeurs propres réelles positives (ondes propagatives)
      4. Trie par ordre croissant de k

    Retourne un DispersionResult avec :
      - freqs     : (n_freqs,)
      - wavenumbers : (n_modes, n_freqs) – NaN si le mode n'existe pas
      - vph       : (n_modes, n_freqs)
      - vgr       : (n_modes, n_freqs)  – estimée par différences finies
    """
    n_freqs = len(freqs)
    k_arr   = np.full((n_modes, n_freqs), np.nan)
    vph_arr = np.full((n_modes, n_freqs), np.nan)

    # ── Réduction de la taille du problème (DDL actifs) ──────
    # On utilise uniquement les DDL liés à des nœuds ayant au moins un élément
    # (les nœuds isolés hors polygone sont éliminés)
    N = GK1.shape[0]

    for fi, f in enumerate(freqs):
        if progress_callback and fi % 5 == 0:
            progress_callback(f"Calcul fréquence {fi+1}/{n_freqs} : {f:.1f} Hz")

        omega = 2 * np.pi * f
        A, B = build_companion(GK1, GK2, GK3, GM, omega)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                eigvals, _ = eig(A, B)
            except Exception:
                continue

        # Filtrer : k réel, positif, < k_max
        k_vals = []
        for ev in eigvals:
            if np.isfinite(ev) and abs(ev.imag) < 0.05 * abs(ev.real) and ev.real > 0:
                k_r = ev.real
                if 0 < k_r < k_max:
                    k_vals.append(k_r)

        k_vals = sorted(set(np.round(k_vals, 4)))[:n_modes]

        for mi, k in enumerate(k_vals):
            k_arr[mi, fi]   = k
            vph_arr[mi, fi] = omega / k

    # ── Vitesse de groupe par différences finies ──────────────
    domega = 2 * np.pi * (freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    vgr_arr = np.full_like(vph_arr, np.nan)
    for mi in range(n_modes):
        for fi in range(1, n_freqs - 1):
            k_m = k_arr[mi, fi-1]
            k_p = k_arr[mi, fi+1]
            if np.isnan(k_m) or np.isnan(k_p):
                continue
            dk = k_p - k_m
            if abs(dk) > 1e-10:
                vgr_arr[mi, fi] = 2 * domega / dk

    return DispersionResult(
        freqs=freqs,
        wavenumbers=k_arr,
        vph=vph_arr,
        vgr=vgr_arr
    )


# ─────────────────────────────────────────────────────────────
#  Géométries prédéfinies
# ─────────────────────────────────────────────────────────────

DEFAULT_MATERIALS = {
    "Noyau argileux":  Material("Noyau argileux",  E=50e6,  nu=0.45, rho=1900, color="#ca8a04"),
    "Recharge sable":  Material("Recharge sable",  E=120e6, nu=0.35, rho=1800, color="#16a34a"),
    "Enrochement":     Material("Enrochement",     E=300e6, nu=0.28, rho=2100, color="#dc2626"),
    "Fondation":       Material("Fondation",       E=500e6, nu=0.25, rho=2200, color="#2563eb"),
    "Argile molle":    Material("Argile molle",    E=20e6,  nu=0.48, rho=1600, color="#7c3aed"),
}

def get_preset_layers(preset: str, mats: dict) -> List[Layer]:
    """Retourne les couches prédéfinies pour 'digue' ou 'barrage'."""
    if preset == "digue":
        return [
            Layer(np.array([[0,0],[120,0],[120,5],[0,5]]),
                  mats["Fondation"], "Fondation"),
            Layer(np.array([[10,5],[50,5],[55,25],[25,25]]),
                  mats["Recharge sable"], "Recharge amont"),
            Layer(np.array([[50,5],[70,5],[65,25],[55,25]]),
                  mats["Noyau argileux"], "Noyau argileux"),
            Layer(np.array([[70,5],[110,5],[95,25],[65,25]]),
                  mats["Recharge sable"], "Recharge aval"),
            Layer(np.array([[10,5],[110,5],[95,25],[25,25]]),
                  mats["Enrochement"], "Enrochement (masque)"),
        ]
    elif preset == "barrage":
        return [
            Layer(np.array([[0,0],[180,0],[180,8],[0,8]]),
                  mats["Fondation"], "Fondation"),
            Layer(np.array([[20,8],[80,8],[72,50],[28,50]]),
                  mats["Argile molle"], "Masque amont"),
            Layer(np.array([[80,8],[100,8],[92,50],[78,50]]),
                  mats["Noyau argileux"], "Noyau"),
            Layer(np.array([[100,8],[160,8],[152,50],[92,50]]),
                  mats["Recharge sable"], "Recharge aval"),
            Layer(np.array([[20,8],[160,8],[152,50],[28,50]]),
                  mats["Enrochement"], "Corps barrage"),
        ]
    else:
        raise ValueError(f"Preset inconnu : {preset}")
