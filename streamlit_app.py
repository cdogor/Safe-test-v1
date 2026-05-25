"""
app.py
======
Application Streamlit – Calcul de modes de dispersion SAFE 2.5D
pour milieu de type digue ou barrage en remblai.

Lancer avec :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import io, json, time
from copy import deepcopy

from safe_engine import (
    Material, Layer, DEFAULT_MATERIALS,
    assemble_global, solve_dispersion,
    get_preset_layers, DispersionResult
)

# ─────────────────────────────────────────────────────────────
#  Config Streamlit
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAFE 2.5D – Dispersion",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e2eaf4; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #111827; border: 1px solid #1e2d45;
        border-radius: 8px; padding: 12px 16px; margin: 4px 0;
    }
    .mode-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 11px; font-family: monospace; margin: 2px;
    }
    h1, h2, h3 { color: #00d4ff !important; }
    .stButton>button {
        background: linear-gradient(135deg, #006699, #00d4ff) !important;
        color: #000 !important; font-weight: bold !important;
        border: none !important; border-radius: 6px !important;
    }
    .stButton>button:hover { opacity: 0.85 !important; }
    .sidebar .stButton>button {
        background: #1e2d45 !important; color: #e2eaf4 !important;
    }
    div[data-testid="stExpander"] { border: 1px solid #1e2d45 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────

def init_state():
    if "materials" not in st.session_state:
        st.session_state.materials = deepcopy(DEFAULT_MATERIALS)
    if "layers" not in st.session_state:
        st.session_state.layers = get_preset_layers("digue", st.session_state.materials)
    if "result" not in st.session_state:
        st.session_state.result = None
    if "mesh_info" not in st.session_state:
        st.session_state.mesh_info = None
    if "log" not in st.session_state:
        st.session_state.log = []

init_state()


# ─────────────────────────────────────────────────────────────
#  Fonctions utilitaires de tracé
# ─────────────────────────────────────────────────────────────

MAT_COLORS = {
    "Noyau argileux": "#ca8a04",
    "Recharge sable": "#16a34a",
    "Enrochement":    "#dc2626",
    "Fondation":      "#2563eb",
    "Argile molle":   "#7c3aed",
}
PALETTE = ["#00d4ff","#ff6b35","#7fff6b","#ff3399","#ffd700","#c084fc","#fb923c","#34d399"]


def fig_geometry(layers: list) -> plt.Figure:
    """Tracé matplotlib de la section transversale avec matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1525")
    ax.set_facecolor("#0d1525")

    legend_handles = {}
    for layer in layers:
        poly = layer.polygon
        patch = MplPolygon(poly, closed=True,
                           facecolor=layer.material.color + "cc",
                           edgecolor="#ffffff44", linewidth=0.8)
        ax.add_patch(patch)
        label = layer.material.name
        if label not in legend_handles:
            legend_handles[label] = mpatches.Patch(
                facecolor=layer.material.color, label=label)
        # Centroïde pour label
        cx, cz = poly.mean(axis=0)
        ax.text(cx, cz, layer.label, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
                fontfamily="monospace")

    ax.autoscale()
    ax.set_xlabel("x (m)", color="#00d4ff", fontfamily="monospace")
    ax.set_ylabel("z (m)", color="#00d4ff", fontfamily="monospace")
    ax.tick_params(colors="#5a7090")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.legend(handles=list(legend_handles.values()), loc="upper right",
              fontsize=8, facecolor="#111827", labelcolor="white",
              edgecolor="#1e2d45")
    ax.set_title("Section transversale – Plan (x, z)", color="#00d4ff",
                 fontfamily="monospace", fontsize=10)
    fig.tight_layout()
    return fig


def fig_dispersion(result: DispersionResult, show_vgr: bool = True,
                   fmin: float = 0, fmax: float = None,
                   vmin: float = 0, vmax: float = None) -> go.Figure:
    """Courbes de dispersion Vph(f) + Vgr(f) avec Plotly."""
    freqs = result.freqs
    mask_f = (freqs >= fmin) & (freqs <= (fmax or freqs.max() + 1))

    rows = 2 if show_vgr else 1
    titles = ["Vitesse de phase Vph(f)", "Vitesse de groupe Vgr(f)"] if show_vgr else ["Vitesse de phase Vph(f)"]
    fig = make_subplots(rows=rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.12)

    for mi in range(result.wavenumbers.shape[0]):
        k_m = result.wavenumbers[mi, mask_f]
        vph_m = result.vph[mi, mask_f]
        vgr_m = result.vgr[mi, mask_f]
        f_m   = freqs[mask_f]

        valid = ~np.isnan(vph_m)
        if not valid.any():
            continue

        color = PALETTE[mi % len(PALETTE)]
        name  = f"Mode {mi+1}"

        fig.add_trace(go.Scatter(
            x=f_m[valid], y=vph_m[valid], name=name,
            mode="lines+markers", marker=dict(size=4),
            line=dict(color=color, width=2),
            legendgroup=name,
        ), row=1, col=1)

        if show_vgr:
            valid_g = ~np.isnan(vgr_m) & (vgr_m > 0) & (vgr_m < 5000)
            if valid_g.any():
                fig.add_trace(go.Scatter(
                    x=f_m[valid_g], y=vgr_m[valid_g], name=name,
                    mode="lines+markers", marker=dict(size=4),
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=name, showlegend=False,
                ), row=2, col=1)

    layout_common = dict(
        paper_bgcolor="#0d1525", plot_bgcolor="#0d1525",
        font=dict(color="#e2eaf4", family="monospace"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45",
                    borderwidth=1, font=dict(size=10)),
    )
    fig.update_layout(height=500 if show_vgr else 320,
                      **layout_common, margin=dict(l=60,r=20,t=40,b=40))
    for i in range(1, rows+1):
        fig.update_xaxes(title_text="Fréquence (Hz)",
                         gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                         range=[fmin, fmax or freqs.max()], row=i, col=1)
    fig.update_yaxes(title_text="Vph (m/s)",
                     gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                     range=[vmin or 0, vmax or None], row=1, col=1)
    if show_vgr:
        fig.update_yaxes(title_text="Vgr (m/s)",
                         gridcolor="#1e2d45", zerolinecolor="#1e2d45",
                         row=2, col=1)
    return fig


def fig_wavenumber(result: DispersionResult, fmax: float = None) -> go.Figure:
    """Diagramme k(f) avec Plotly."""
    freqs = result.freqs
    mask_f = freqs <= (fmax or freqs.max() + 1)

    fig = go.Figure()
    for mi in range(result.wavenumbers.shape[0]):
        k_m = result.wavenumbers[mi, mask_f]
        f_m = freqs[mask_f]
        valid = ~np.isnan(k_m)
        if not valid.any():
            continue
        fig.add_trace(go.Scatter(
            x=f_m[valid], y=k_m[valid],
            name=f"Mode {mi+1}",
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(color=PALETTE[mi % len(PALETTE)], width=1.8),
        ))

    fig.update_layout(
        title="Diagramme k(f) – nombres d'onde propagatifs",
        xaxis_title="Fréquence (Hz)", yaxis_title="k (rad/m)",
        paper_bgcolor="#0d1525", plot_bgcolor="#0d1525",
        font=dict(color="#e2eaf4", family="monospace"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=40), height=320,
    )
    fig.update_xaxes(gridcolor="#1e2d45")
    fig.update_yaxes(gridcolor="#1e2d45")
    return fig


def fig_wavelength(result: DispersionResult, fmax: float = None) -> go.Figure:
    """Courbes de longueur d'onde λ(f)."""
    freqs = result.freqs
    mask_f = freqs <= (fmax or freqs.max() + 1)

    fig = go.Figure()
    for mi in range(result.wavenumbers.shape[0]):
        k_m = result.wavenumbers[mi, mask_f]
        f_m = freqs[mask_f]
        valid = ~np.isnan(k_m) & (k_m > 0)
        if not valid.any():
            continue
        lam_m = 2 * np.pi / k_m[valid]
        fig.add_trace(go.Scatter(
            x=f_m[valid], y=lam_m,
            name=f"Mode {mi+1}",
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(color=PALETTE[mi % len(PALETTE)], width=1.8),
        ))

    fig.update_layout(
        title="Longueur d'onde λ(f)",
        xaxis_title="Fréquence (Hz)", yaxis_title="λ (m)",
        paper_bgcolor="#0d1525", plot_bgcolor="#0d1525",
        font=dict(color="#e2eaf4", family="monospace"),
        legend=dict(bgcolor="#111827", bordercolor="#1e2d45", borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=40), height=300,
    )
    fig.update_xaxes(gridcolor="#1e2d45")
    fig.update_yaxes(gridcolor="#1e2d45")
    return fig


def result_to_dataframe(result: DispersionResult) -> pd.DataFrame:
    """Exporte les résultats en DataFrame."""
    rows = []
    for mi in range(result.wavenumbers.shape[0]):
        for fi, f in enumerate(result.freqs):
            k   = result.wavenumbers[mi, fi]
            vph = result.vph[mi, fi]
            vgr = result.vgr[mi, fi]
            if not np.isnan(k):
                rows.append({
                    "Mode":       mi + 1,
                    "f (Hz)":     round(f, 3),
                    "k (rad/m)":  round(k, 4),
                    "λ (m)":      round(2*np.pi/k, 3) if k > 0 else np.nan,
                    "Vph (m/s)":  round(vph, 2),
                    "Vgr (m/s)":  round(vgr, 2) if not np.isnan(vgr) else np.nan,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
#  Sidebar – Paramètres
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Paramètres SAFE")

    # ── Preset géométrie ─────────────────────────────────────
    st.markdown("### 🏔 Géométrie")
    preset = st.selectbox("Structure", ["digue", "barrage"],
                          format_func=lambda x: "Digue en remblai" if x=="digue" else "Barrage en remblai")
    if st.button("Charger preset"):
        st.session_state.layers = get_preset_layers(preset, st.session_state.materials)
        st.session_state.result = None
        st.rerun()

    st.divider()

    # ── Matériaux ────────────────────────────────────────────
    st.markdown("### 🪨 Matériaux")
    mat_names = list(st.session_state.materials.keys())
    sel_mat = st.selectbox("Modifier matériau", mat_names)
    mat = st.session_state.materials[sel_mat]

    with st.expander(f"✏ {sel_mat}", expanded=False):
        new_E   = st.number_input("Module de Young E (Pa)",   value=float(mat.E),   step=1e6, format="%.3e")
        new_nu  = st.number_input("Coefficient de Poisson ν", value=float(mat.nu),  step=0.01, min_value=0.01, max_value=0.49)
        new_rho = st.number_input("Densité ρ (kg/m³)",        value=float(mat.rho), step=10.0)
        if st.button("Appliquer"):
            st.session_state.materials[sel_mat] = Material(sel_mat, new_E, new_nu, new_rho,
                                                           mat.color)
            st.session_state.result = None
            st.rerun()

        st.markdown(f"""
        <div class='metric-card'>
        Vs = <b>{mat.Vs:.0f}</b> m/s &nbsp;|&nbsp; Vp = <b>{mat.Vp:.0f}</b> m/s
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Paramètres calcul ────────────────────────────────────
    st.markdown("### 🔢 Calcul")
    nx = st.slider("Résolution maillage nx", 4, 16, 6,
                   help="Éléments par couche dans la direction x")
    nz = st.slider("Résolution maillage nz", 3, 12, 4,
                   help="Éléments par couche dans la direction z")
    f_min  = st.number_input("f min (Hz)", value=5.0,  step=1.0)
    f_max  = st.number_input("f max (Hz)", value=150.0, step=10.0)
    n_freq = st.slider("Nombre de fréquences", 20, 150, 60)
    n_modes= st.slider("Nombre de modes max",   4,  12,  6)
    k_max  = st.number_input("k max (rad/m)", value=200.0, step=10.0)

    st.divider()
    st.markdown("### 🖼 Affichage")
    show_vgr = st.checkbox("Afficher Vgr(f)", value=True)
    show_kf  = st.checkbox("Afficher k(f)",   value=True)
    show_lam = st.checkbox("Afficher λ(f)",   value=True)
    vph_min  = st.number_input("Vph min (m/s)", value=0.0, step=50.0)
    vph_max  = st.number_input("Vph max (m/s)", value=0.0, step=50.0,
                                help="0 = automatique")

    st.divider()
    if st.button("🗑 Réinitialiser"):
        st.session_state.result = None
        st.session_state.log = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  Corps principal
# ─────────────────────────────────────────────────────────────

st.markdown("# 🌊 SAFE 2.5D — Dispersion Multicomposante")
st.markdown(
    "<small style='color:#5a7090'>Semi-Analytical Finite Element Method · "
    "Éléments Q4 · Solveur quadratique aux valeurs propres · "
    "Milieu type digue/barrage en remblai</small>",
    unsafe_allow_html=True
)
st.divider()

# ── Onglets principaux ────────────────────────────────────────
tab_geo, tab_mat, tab_calc, tab_disp, tab_export = st.tabs([
    "📐 Géométrie", "🪨 Matériaux", "⚙ Calcul", "📊 Dispersion", "💾 Export"
])


# ──────────── Onglet GÉOMÉTRIE ──────────────────────────────
with tab_geo:
    st.subheader("Section transversale")
    col1, col2 = st.columns([3, 1])

    with col1:
        fig_geo = fig_geometry(st.session_state.layers)
        st.pyplot(fig_geo, use_container_width=True)
        plt.close(fig_geo)

    with col2:
        st.markdown("**Couches actives**")
        for i, layer in enumerate(st.session_state.layers):
            color = layer.material.color
            st.markdown(
                f"<div class='metric-card'>"
                f"<span style='color:{color}'>■</span> "
                f"<b>Couche {i}</b> : {layer.label}<br>"
                f"<small style='color:#5a7090'>{layer.material.name}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Édition des polygones
    with st.expander("✏ Modifier les polygones (JSON)"):
        st.info("Format : liste de couches, chaque couche = {'label', 'material', 'polygon' [[x,z],...]}")
        layers_json = json.dumps([
            {"label": l.label,
             "material": l.material.name,
             "polygon": l.polygon.tolist()}
            for l in st.session_state.layers
        ], indent=2)
        edited = st.text_area("Couches (JSON)", layers_json, height=300)
        if st.button("Appliquer JSON"):
            try:
                data = json.loads(edited)
                new_layers = []
                for d in data:
                    mat = st.session_state.materials.get(d["material"])
                    if mat is None:
                        st.error(f"Matériau inconnu : {d['material']}")
                        break
                    new_layers.append(Layer(
                        polygon=np.array(d["polygon"], dtype=float),
                        material=mat,
                        label=d["label"]
                    ))
                else:
                    st.session_state.layers = new_layers
                    st.session_state.result = None
                    st.success("Couches mises à jour ✓")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur JSON : {e}")


# ──────────── Onglet MATÉRIAUX ──────────────────────────────
with tab_mat:
    st.subheader("Propriétés des matériaux")
    cols = st.columns(len(st.session_state.materials))
    for col, (name, mat) in zip(cols, st.session_state.materials.items()):
        with col:
            color = mat.color
            st.markdown(
                f"<div class='metric-card' style='border-color:{color}55'>"
                f"<b style='color:{color}'>{name}</b><hr style='border-color:{color}33'>"
                f"E = {mat.E:.2e} Pa<br>"
                f"ν = {mat.nu}<br>"
                f"ρ = {mat.rho} kg/m³<br>"
                f"<b>Vs = {mat.Vs:.0f} m/s</b><br>"
                f"<b>Vp = {mat.Vp:.0f} m/s</b>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # Tableau récapitulatif
    df_mat = pd.DataFrame([
        {
            "Matériau": name,
            "E (MPa)":  round(mat.E / 1e6, 1),
            "ν":        mat.nu,
            "ρ (kg/m³)": mat.rho,
            "Vs (m/s)": round(mat.Vs),
            "Vp (m/s)": round(mat.Vp),
            "λ_Lamé (MPa)": round(mat.lam / 1e6, 1),
            "μ (MPa)":      round(mat.mu / 1e6, 1),
        }
        for name, mat in st.session_state.materials.items()
    ])
    st.dataframe(df_mat, use_container_width=True, hide_index=True)

    # Ajouter un matériau personnalisé
    with st.expander("➕ Ajouter un matériau personnalisé"):
        c1, c2, c3, c4, c5 = st.columns(5)
        nm = c1.text_input("Nom")
        eM = c2.number_input("E (Pa)", value=100e6, format="%.2e")
        nM = c3.number_input("ν", value=0.30, step=0.01)
        rM = c4.number_input("ρ (kg/m³)", value=1800.0)
        cM = c5.color_picker("Couleur", "#888888")
        if st.button("Ajouter matériau") and nm:
            st.session_state.materials[nm] = Material(nm, eM, nM, rM, cM)
            st.rerun()


# ──────────── Onglet CALCUL ─────────────────────────────────
with tab_calc:
    st.subheader("Lancer le calcul SAFE")

    # Résumé des paramètres
    n_elem_est = nx * nz * len(st.session_state.layers)
    n_dof_est  = 3 * (nx + 1) * (nz + 1) * len(st.session_state.layers)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Couches", len(st.session_state.layers))
    col2.metric("Éléments estimés", n_elem_est)
    col3.metric("DDL estimés", n_dof_est)
    col4.metric("Fréquences", n_freq)

    st.info(
        f"**Formulation :** K₁ + i·k·K₂ + k²·K₃ – ω²·M = 0  "
        f"→ problème quadratique en k de taille {n_dof_est}×{n_dof_est}  "
        f"→ résolu par mise en forme compagnon avec **scipy.linalg.eig**"
    )

    # Avertissement taille
    if n_dof_est > 500:
        st.warning(
            f"⚠ {n_dof_est} DDL — le calcul peut prendre quelques minutes. "
            "Réduire nx/nz ou le nombre de fréquences pour accélérer."
        )

    log_container = st.empty()
    progress_bar  = st.empty()

    if st.button("▶ LANCER LE CALCUL SAFE", use_container_width=True):
        st.session_state.log = []
        logs = []

        def log(msg):
            logs.append(msg)
            log_container.text_area("Log", "\n".join(logs), height=150)

        def progress(msg):
            log(msg)

        t0 = time.time()

        with st.spinner("Assemblage des matrices…"):
            try:
                log("▶ Assemblage des matrices élémentaires SAFE…")
                progress_bar.progress(0.1)
                mesh, GK1, GK2, GK3, GM = assemble_global(
                    st.session_state.layers, nx=nx, nz=nz,
                    progress_callback=progress
                )
                st.session_state.mesh_info = {
                    "n_nodes": len(mesh.nodes),
                    "n_elem":  len(mesh.connectivity),
                    "n_dof":   mesh.n_dof
                }
                log(f"✓ Assemblage terminé : {mesh.n_dof} DDL")
                progress_bar.progress(0.3)
            except Exception as e:
                st.error(f"Erreur assemblage : {e}")
                st.stop()

        with st.spinner("Calcul des courbes de dispersion…"):
            try:
                freqs = np.linspace(f_min, f_max, n_freq)
                log(f"▶ Balayage {n_freq} fréquences [{f_min:.0f}–{f_max:.0f} Hz]…")
                progress_bar.progress(0.4)

                result = solve_dispersion(
                    GK1, GK2, GK3, GM,
                    freqs=freqs,
                    n_modes=n_modes,
                    k_max=k_max,
                    progress_callback=progress
                )
                st.session_state.result = result
                progress_bar.progress(1.0)

                dt = time.time() - t0
                n_pts = int(np.sum(~np.isnan(result.wavenumbers)))
                log(f"✓ Calcul terminé en {dt:.1f}s — {n_pts} points de dispersion")
                st.success(f"✅ Calcul terminé en {dt:.1f}s — {n_pts} points calculés sur {n_modes} modes")

            except Exception as e:
                st.error(f"Erreur calcul dispersion : {e}")
                raise

    # Affichage info maillage
    if st.session_state.mesh_info:
        mi = st.session_state.mesh_info
        st.markdown(
            f"<div class='metric-card'>Maillage actif : "
            f"<b>{mi['n_nodes']}</b> nœuds · "
            f"<b>{mi['n_elem']}</b> éléments Q4 · "
            f"<b>{mi['n_dof']}</b> DDL</div>",
            unsafe_allow_html=True
        )


# ──────────── Onglet DISPERSION ──────────────────────────────
with tab_disp:
    st.subheader("Courbes de dispersion")

    if st.session_state.result is None:
        st.info("Lancez le calcul dans l'onglet **Calcul** pour afficher les résultats.")
    else:
        result = st.session_state.result
        freqs  = result.freqs
        fmax_d = float(freqs.max())

        # Statistiques des modes
        st.markdown("**Modes détectés**")
        mode_cols = st.columns(min(n_modes, 8))
        for mi in range(n_modes):
            k_m   = result.wavenumbers[mi]
            vph_m = result.vph[mi]
            valid = ~np.isnan(vph_m)
            if not valid.any():
                continue
            with mode_cols[mi % len(mode_cols)]:
                st.markdown(
                    f"<div class='metric-card' style='border-color:{PALETTE[mi%8]}55'>"
                    f"<b style='color:{PALETTE[mi%8]}'>Mode {mi+1}</b><br>"
                    f"Vph: {vph_m[valid].min():.0f}–{vph_m[valid].max():.0f} m/s<br>"
                    f"<small>{valid.sum()} pts</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # Vph(f) + Vgr(f)
        vmax_d = vph_max if vph_max > 0 else None
        st.plotly_chart(
            fig_dispersion(result, show_vgr=show_vgr,
                           fmin=0, fmax=fmax_d,
                           vmin=vph_min, vmax=vmax_d),
            use_container_width=True
        )

        # k(f)
        if show_kf:
            st.plotly_chart(fig_wavenumber(result, fmax=fmax_d),
                            use_container_width=True)

        # λ(f)
        if show_lam:
            st.plotly_chart(fig_wavelength(result, fmax=fmax_d),
                            use_container_width=True)

        # Tableau interactif
        with st.expander("📋 Tableau des valeurs"):
            df = result_to_dataframe(result)
            st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────── Onglet EXPORT ─────────────────────────────────
with tab_export:
    st.subheader("Export des résultats")

    if st.session_state.result is None:
        st.info("Aucun résultat disponible. Lancez le calcul d'abord.")
    else:
        result = st.session_state.result
        df = result_to_dataframe(result)

        # CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Télécharger CSV (dispersion)",
            csv_bytes, "safe_dispersion.csv", "text/csv",
            use_container_width=True
        )

        # JSON complet
        export_dict = {
            "freqs": result.freqs.tolist(),
            "wavenumbers": np.where(np.isnan(result.wavenumbers), None,
                                    result.wavenumbers).tolist(),
            "vph": np.where(np.isnan(result.vph), None, result.vph).tolist(),
            "vgr": np.where(np.isnan(result.vgr), None, result.vgr).tolist(),
            "materials": {
                name: {"E": m.E, "nu": m.nu, "rho": m.rho}
                for name, m in st.session_state.materials.items()
            },
            "layers": [
                {"label": l.label, "material": l.material.name,
                 "polygon": l.polygon.tolist()}
                for l in st.session_state.layers
            ]
        }
        json_bytes = json.dumps(export_dict, indent=2).encode("utf-8")
        st.download_button(
            "⬇ Télécharger JSON (résultats complets)",
            json_bytes, "safe_results.json", "application/json",
            use_container_width=True
        )

        # PNG des courbes Vph
        st.markdown("**Aperçu figure exportable**")
        fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
        fig_exp.patch.set_facecolor("#0d1525")
        ax_exp.set_facecolor("#0d1525")
        for mi in range(result.wavenumbers.shape[0]):
            vph_m = result.vph[mi]
            valid = ~np.isnan(vph_m)
            if not valid.any():
                continue
            ax_exp.plot(result.freqs[valid], vph_m[valid],
                        color=PALETTE[mi % len(PALETTE)],
                        linewidth=2, marker="o", markersize=3,
                        label=f"Mode {mi+1}")
        ax_exp.set_xlabel("Fréquence (Hz)", color="#00d4ff", fontfamily="monospace")
        ax_exp.set_ylabel("Vph (m/s)", color="#00d4ff", fontfamily="monospace")
        ax_exp.set_title("SAFE 2.5D – Courbes de dispersion Vph(f)",
                         color="#00d4ff", fontfamily="monospace")
        ax_exp.tick_params(colors="#5a7090")
        ax_exp.legend(facecolor="#111827", labelcolor="white", edgecolor="#1e2d45")
        ax_exp.grid(color="#1e2d45", linewidth=0.5)
        for spine in ax_exp.spines.values():
            spine.set_edgecolor("#1e2d45")
        fig_exp.tight_layout()

        buf = io.BytesIO()
        fig_exp.savefig(buf, format="png", dpi=150, facecolor="#0d1525")
        plt.close(fig_exp)
        buf.seek(0)
        st.image(buf, use_container_width=True)
        buf.seek(0)
        st.download_button(
            "⬇ Télécharger PNG",
            buf, "safe_dispersion.png", "image/png",
            use_container_width=True
        )

# ── Pied de page ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<small style='color:#1e2d45'>"
    "SAFE 2.5D · Q4 Gauss 2×2 · Formulation Bartoli et al. (2006) · "
    "scipy.linalg.eig · Hayashi & Rose (2003) · "
    "Castaings & Lowe (2008)"
    "</small>",
    unsafe_allow_html=True
)
