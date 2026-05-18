# =============================================================================
# Atmospheric-Ocean Systems Visualization Dashboard
# IEEE SciVis Contest 2026 — Task 2: LLC2160 Ocean Data
# =============================================================================
#
# INSTALLATION (copy-paste into terminal):
#   pip install streamlit plotly numpy pandas OpenVisus
#
# RUN:
#   streamlit run app.py
#
# REQUIREMENTS:
#   Python 3.8–3.11  |  Internet access to NSDF server
#   Host: nsdf-climate3-origin.nationalresearchplatform.org:50098
#
# DATA SOURCE:
#   NASA DYAMOND LLC2160 Ocean Simulation
#   Hosted by NSDF / National Research Platform via OpenVisus IDX
#   Reference: https://github.com/sci-visus/sciviscontest2026
#   Notebook:  notebooks_examples/ieee_scivis_dyamond_ocean.ipynb
#
# KNOWN URL PATTERNS (from reference notebook):
#   salt  → mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx
#   Theta → mit_output/llc2160_theta/llc2160_theta.idx
#   u     → mit_output/llc2160_arco/visus.idx   (ARCO multi-field)
#   v     → mit_output/llc2160_v/v_llc2160_x_y_depth.idx
#   w     → mit_output/llc2160_w/llc2160_w.idx
#
# EXTENDING THE DASHBOARD — quick guide:
#   • Add a variable  → DATASET_URLS + VARIABLE_META (no other changes)
#   • Add a new plot  → write create_*_figure(), register in PLOT_REGISTRY,
#                       add a checkbox in render_sidebar() and call
#                       render_plot_box() in the main UI section.
# =============================================================================

from __future__ import annotations

import datetime
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy dependencies — handled gracefully
# ─────────────────────────────────────────────────────────────────────────────
try:
    import OpenVisus as ov  # pip install OpenVisus
    OPENVISUS_AVAILABLE = True
except ImportError:
    OPENVISUS_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# OpenVisus local tile cache (speeds up repeated access)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".visus-cache", "dyamond-ocean")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("VISUS_CACHE", _CACHE_DIR)

# Directorio de slices pre-descargados por download_data.py
_SLICE_DIR = os.path.join(_PROJECT_ROOT, ".visus-cache", "slices")
os.makedirs(_SLICE_DIR, exist_ok=True)

# ── LLC2160 Ocean Dataset URLs ─────────────────────────────────────────────
# Storage: NSDF National Research Platform
# Server:  nsdf-climate3-origin.nationalresearchplatform.org:50098
# Format:  OpenVisus IDX
#
# IMPORTANT — each variable uses a DIFFERENT path pattern (from reference notebook):
#   salt, theta, P, V  →  mit_output/llc2160_<var>/<var>_llc2160_x_y_depth.idx
#   w, theta (alt)     →  mit_output/llc2160_<var>/llc2160_<var>.idx
#   u                  →  mit_output/llc2160_arco/visus.idx
#
# TO ADD A NEW VARIABLE:
#   1. Add an entry to DATASET_URLS (key = field name used by db.read())
#   2. Add matching metadata to VARIABLE_META
#   Done — no other changes needed.

_NSDF_BASE = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

DATASET_URLS: Dict[str, str] = {
    # salt  → mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx
    "salt":  _NSDF_BASE + "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
    # Theta → mit_output/llc2160_theta/llc2160_theta.idx  (special case like w)
    "Theta": _NSDF_BASE + "mit_output/llc2160_theta/llc2160_theta.idx",
    # u     → mit_output/llc2160_arco/visus.idx           (ARCO multi-field)
    "u":     _NSDF_BASE + "mit_output/llc2160_arco/visus.idx",
    # v     → mit_output/llc2160_v/v_llc2160_x_y_depth.idx
    "v":     _NSDF_BASE + "mit_output/llc2160_v/v_llc2160_x_y_depth.idx",
    # w     → mit_output/llc2160_w/llc2160_w.idx
    "w":     _NSDF_BASE + "mit_output/llc2160_w/llc2160_w.idx",
}

# Variable display metadata — label, units, default colour range, Plotly colorscale
VARIABLE_META: Dict[str, Dict[str, Any]] = {
    "salt":  {"label": "Sea Surface Salinity",  "units": "g kg⁻¹", "cmap": "haline",  "vmin": 31.0,  "vmax": 38.0},
    "Theta": {"label": "Potential Temperature", "units": "°C",     "cmap": "thermal", "vmin": -2.0,  "vmax": 32.0},
    "u":     {"label": "Zonal Velocity (ARCO)", "units": "m/s",    "cmap": "balance", "vmin": -1.5,  "vmax": 1.5},
    "v":     {"label": "Meridional Velocity",   "units": "m/s",    "cmap": "balance", "vmin": -1.5,  "vmax": 1.5},
    "w":     {"label": "Vertical Velocity",     "units": "m/s",    "cmap": "delta",   "vmin": -0.01, "vmax": 0.01},
}

# LLC2160 dataset physical constants
LLC2160_TOTAL_TIMESTEPS: int = 10000  # ~14 months hourly from 2020-01-20
LLC2160_DEPTH_LEVELS:    int = 90
_T0 = datetime.datetime(2020, 1, 20, 0, 0)  # simulation start

# OpenVisus resolution quality levels (-6 = coarsest/fastest, 0 = full)
# MEMORY GUIDE (approximate output size for global view):
#   -6 → ~128×96   px  ~  0.05 MB  ✅ safe
#   -5 → ~256×192  px  ~  0.2  MB  ✅ safe
#   -4 → ~512×384  px  ~  0.8  MB  ✅ safe  ← default
#   -3 → ~1024×768 px  ~  3    MB  ✅ safe for small regions
#   -2 → ~2048×...  px ~  12   MB  ⚠️  use only for small regions
#   -1 → full res       ~  800+ MB  ❌ will crash browser for global view
QUALITY_OPTIONS: Dict[str, int] = {
    "Very Fast (~128 px)":  -6,
    "Fast (~256 px)":       -5,
    "Medium (~512 px)":     -4,
    "Good (~1k px, slow)":  -3,
}
DEFAULT_QUALITY_KEY = "Fast (~256 px)"

# Hard cap: never send more than this many pixels to the browser per axis.
# Protects against OOM crashes regardless of quality setting.
MAX_RENDER_PX: int = 800

# Preset geographic regions  (lat_min, lat_max, lon_min, lon_max)
GEO_PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "Global":         (-90,  90, -180, 180),
    "North Atlantic": (  0,  70,  -80,  20),
    "South Atlantic": (-60,   0,  -70,  20),
    "Pacific":        (-60,  60,  120, -70),
    "Indian Ocean":   (-60,  30,   20, 120),
    "Arctic":         ( 60,  90, -180, 180),
    "Antarctic":      (-90, -60, -180, 180),
    "Mediterranean":  ( 30,  48,   -5,  42),
    "Gulf of Mexico": ( 18,  31,  -98, -80),
    "Custom":         None,
}

# =============================================================================
# SECTION 2 — DATA LAYER
# =============================================================================

@st.cache_resource(show_spinner=False)
def _connect_dataset(variable: str) -> Optional[Any]:
    """
    Open a persistent OpenVisus connection for a given ocean variable.

    Cached as a Streamlit resource (one connection object per variable per
    process lifetime). Returns None when OpenVisus is unavailable or the
    remote dataset cannot be reached.

    Parameters
    ----------
    variable : str
        Ocean field name (key in DATASET_URLS).

    Returns
    -------
    OpenVisus PyDataset object or None.
    """
    if not OPENVISUS_AVAILABLE:
        return None
    url = DATASET_URLS.get(variable)
    if not url:
        return None
    try:
        return ov.LoadDataset(url)
    except Exception as exc:
        logging.warning("OpenVisus connection failed for '%s': %s", variable, exc)
        return None



# =============================================================================
# SECCION 2b — DISCO-CACHE
# Generado por download_data.py. La clave es simple (no usa hash de floats).
# =============================================================================

def _slice_filename(variable: str, timestep: int, depth: int, quality: int) -> str:
    """Nombre de archivo — DEBE coincidir con download_data.py."""
    return f"{variable}_t{timestep:05d}_d{depth}_q{quality}_global.npz"


def _disk_load(variable: str, timestep: int, depth: int,
               quality: int) -> Optional[np.ndarray]:
    """Carga un slice global del disco. None si no existe."""
    path = os.path.join(_SLICE_DIR, _slice_filename(variable, timestep, depth, quality))
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=False)["arr"].astype(np.float32)
    except Exception as exc:
        logging.warning("disco-cache corrupto: %s", exc)
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _disk_save(variable: str, timestep: int, depth: int,
               quality: int, arr: np.ndarray) -> None:
    """Guarda un slice en disco (escritura atomica)."""
    path = os.path.join(_SLICE_DIR, _slice_filename(variable, timestep, depth, quality))
    if os.path.exists(path):
        return
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, arr=arr)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _crop_to_region(
    arr: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> np.ndarray:
    """
    Recorta un array global [H, W] a la region normalizada pedida.
    Mucho mas rapido que re-descargar desde la red.
    """
    H, W = arr.shape
    x0 = max(0, int(x_range[0] * W))
    x1 = min(W, int(x_range[1] * W))
    y0 = max(0, int(y_range[0] * H))
    y1 = min(H, int(y_range[1] * H))
    if x1 <= x0: x1 = min(x0 + 1, W)
    if y1 <= y0: y1 = min(y0 + 1, H)
    return arr[y0:y1, x0:x1]


@st.cache_data(show_spinner=False, ttl=300)
def fetch_ocean_slice(
    variable:  str,
    timestep:  int,
    depth:     int,
    quality:   int,
    x_range:   Tuple[float, float],   # normalised lon range [0,1]
    y_range:   Tuple[float, float],   # normalised lat range [0,1]
) -> Optional[np.ndarray]:
    """
    Download a 2-D horizontal ocean field slice from the LLC2160 dataset.

    KEY FIX — spatial subsetting to prevent OOM crashes
    ────────────────────────────────────────────────────
    Calling db.read() without spatial bounds downloads the entire grid
    (~17 280 × 12 960 px at full resolution) which easily consumes several
    GB of RAM and kills the browser tab.

    Instead we:
      1. Query db.getLogicBox() to learn the full pixel dimensions (W × H).
      2. Convert the normalised lat/lon ranges to pixel coordinates.
      3. Pass x=[x0,x1], y=[y0,y1] to db.read() — OpenVisus will stream
         ONLY the requested tile, not the whole dataset.
      4. After reading, apply a hard pixel cap (MAX_RENDER_PX) by downsampling
         with numpy stride slicing to ensure Plotly never gets a grid larger
         than ~800×800 px regardless of quality level.

    Parameters
    ----------
    variable  : Ocean field name, e.g. 'salt'.
    timestep  : Integer time index (0 … LLC2160_TOTAL_TIMESTEPS-1).
    depth     : Depth level index (0 = sea surface).
    quality   : OpenVisus resolution level (-6 = coarsest, -3 = good).
    x_range   : Normalised (lon_min, lon_max) in [0, 1].
    y_range   : Normalised (lat_min, lat_max) in [0, 1].

    Returns
    -------
    2-D float32 numpy array [ny, nx], or None on failure.
    """
    # ── Capa 1: disco-cache (pre-descargado por download_data.py) ──────────
    # Clave simple: solo variable+timestep+depth+quality, SIN region.
    # El slice global se recorta en memoria con _crop_to_region().
    _global = _disk_load(variable, timestep, depth, quality)
    if _global is not None:
        arr = _crop_to_region(_global, x_range, y_range)
        # Aplicar cap de pixeles igual que en la descarga por red
        ny, nx = arr.shape
        sy = max(1, ny // MAX_RENDER_PX)
        sx = max(1, nx // MAX_RENDER_PX)
        if sy > 1 or sx > 1:
            arr = arr[::sy, ::sx]
        return arr

    # ── Capa 2: descarga en tiempo real (fallback cuando no hay cache) ──────
    db = _connect_dataset(variable)
    if db is None:
        return None

    try:
        # ── Step 1: get full logical box dimensions ──────────────────────
        logic_box = db.getLogicBox()          # [[x0,y0,z0], [W,H,D]]
        W = int(logic_box[1][0])              # full width  in pixels
        H = int(logic_box[1][1])              # full height in pixels

        # ── Step 2: convert normalised ranges to pixel indices ───────────
        x0 = max(0,   int(x_range[0] * W))
        x1 = min(W,   int(x_range[1] * W))
        y0 = max(0,   int(y_range[0] * H))
        y1 = min(H,   int(y_range[1] * H))

        # Guard: ensure at least a 1-pixel wide box
        if x1 <= x0:
            x1 = min(x0 + 1, W)
        if y1 <= y0:
            y1 = min(y0 + 1, H)

        # ── Step 3: depth slice as integer indices ───────────────────────
        z_lo = int(depth)
        z_hi = int(depth) + 1

        # ── Step 4: read ONLY the requested spatial tile ─────────────────
        raw = db.read(
            time    = timestep,
            quality = quality,
            x       = [x0, x1],
            y       = [y0, y1],
            z       = [z_lo, z_hi],
        )

        if raw is None:
            return None

        # Squeeze to 2-D (remove singleton depth and any extra leading axes)
        arr = np.array(raw, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr[0]

        # ── Step 5: guardar en disco para futuras sesiones ──────────────
        # Guardamos el array GLOBAL (antes del recorte de region) para que
        # cualquier region posterior pueda reutilizarlo sin re-descargar.
        _disk_save(variable, timestep, depth, quality, arr)

        # ── Step 6: recortar a la region pedida y aplicar cap de pixeles ─
        arr = _crop_to_region(arr, x_range, y_range)
        ny, nx = arr.shape
        stride_y = max(1, ny // MAX_RENDER_PX)
        stride_x = max(1, nx // MAX_RENDER_PX)
        if stride_y > 1 or stride_x > 1:
            arr = arr[::stride_y, ::stride_x]

        return arr

    except Exception as exc:
        logging.warning("read failed (var=%s, t=%d): %s", variable, timestep, exc)
        logging.debug(traceback.format_exc())
        return None


# =============================================================================
# SECTION 3 — SYNTHETIC DEMO DATA
# Used only when OpenVisus is unavailable or the server is unreachable.
# =============================================================================

def _generate_demo_slice(
    variable: str,
    timestep: int,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    n: int = 256,
) -> np.ndarray:
    """
    Return a physically plausible synthetic 2-D field for demonstration.

    The synthetic field is constructed from trigonometric functions that
    approximate the large-scale structure of each variable (e.g., high
    salinity in subtropical gyres, low near poles and equator).

    Parameters
    ----------
    variable  : Ocean field name.
    timestep  : Controls a slow temporal phase shift.
    lat_range : (lat_min, lat_max) in degrees.
    lon_range : (lon_min, lon_max) in degrees.
    n         : Grid resolution (n × n output).

    Returns
    -------
    2-D float32 array [n, n].
    """
    np.random.seed(timestep % 1000 + hash(variable) % 1000)

    lat = np.linspace(lat_range[0], lat_range[1], n)
    lon = np.linspace(lon_range[0], lon_range[1], n)
    LON, LAT = np.meshgrid(lon, lat)
    phase = timestep * 0.01

    meta = VARIABLE_META.get(variable, VARIABLE_META["salt"])
    vmin = meta["vmin"] if meta["vmin"] is not None else -1.0
    vmax = meta["vmax"] if meta["vmax"] is not None else  1.0

    if variable == "salt":
        # Salinity high in subtropics, low near equator and poles
        data = (
            35.0
            + 2.0 * np.sin(np.radians(2.0 * LAT))
            - 1.5 * np.exp(-((LAT / 12.0) ** 2))
            - 1.2 * np.exp(-(((np.abs(LAT) - 70.0) / 6.0) ** 2))
            + 0.6 * np.sin(np.radians(LON) + phase)
            + 0.25 * np.random.randn(n, n)
        )

    elif variable == "Theta":
        # SST peaks in tropics, decreases toward poles
        data = (
            15.0
            - 0.22 * np.abs(LAT)
            + 4.0 * np.cos(np.radians(LAT))
            + 1.5 * np.sin(np.radians(LON * 1.5) + phase)
            + 0.4 * np.random.randn(n, n)
        )

    else:
        # Generic diverging field (velocities, pressure)
        mid = (vmin + vmax) / 2.0
        amp = (vmax - vmin) / 4.0
        data = (
            mid
            + amp * np.sin(np.radians(LON) + phase) * np.cos(np.radians(LAT * 2.0))
            + 0.08 * amp * np.random.randn(n, n)
        )

    return np.clip(data, vmin, vmax).astype(np.float32)


# =============================================================================
# SECTION 4 — COORDINATE UTILITIES
# =============================================================================

def latlon_to_norm(lat: float, lon: float) -> Tuple[float, float]:
    """
    Map geographic (lat, lon) degrees → normalised OpenVisus pixel coords (x, y) ∈ [0, 1].

    LLC2160 grid layout (from reference notebook):
      x-axis = longitude:  -180° … +180°  →  0 … 1
      y-axis = latitude:    -90° …  +90°  →  0 … 1
    """
    x = (lon + 180.0) / 360.0   # longitude → x
    y = (lat +  90.0) / 180.0   # latitude  → y
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def make_axes(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    ny: int, nx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (latitude_1d, longitude_1d) arrays for the given bounds."""
    lats = np.linspace(lat_min, lat_max, ny)
    lons = np.linspace(lon_min, lon_max, nx)
    return lats, lons


def approx_datetime(timestep: int) -> str:
    """Return an approximate UTC datetime string for a given timestep."""
    dt = _T0 + datetime.timedelta(hours=int(timestep))
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# =============================================================================
# SECTION 5 — VISUALISATION LAYER
# =============================================================================
# Each create_*_figure() function is a self-contained plot factory.
#
# TO ADD A NEW PLOT:
#   1. Write `create_my_plot(data, lats, lons, variable, title) -> go.Figure`.
#   2. Add an entry to PLOT_REGISTRY below.
#   3. Add a checkbox in `render_sidebar()` and a call to `render_plot_box()`
#      in the "Optional plots" block inside `main()`.
#   No other changes are required.
# =============================================================================

def _mask_fill(arr: np.ndarray) -> np.ndarray:
    """Replace ocean-model fill values (|v| > 1e6) with NaN."""
    out = arr.astype(np.float64)
    out[np.abs(out) > 1e6] = np.nan
    return out


def _get_clim(arr: np.ndarray, meta: Dict[str, Any]) -> Tuple[float, float]:
    """Return colour limits: use metadata defaults, fall back to data percentiles."""
    vmin = meta.get("vmin")
    vmax = meta.get("vmax")
    valid = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.nanpercentile(valid, 2)) if valid.size else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(valid, 98)) if valid.size else 1.0
    return float(vmin), float(vmax)


def _wrap_longitude_view(
    data: np.ndarray,
    lons: np.ndarray,
    tiles: int = 3,
    period: float = 360.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat the map horizontally so panning across longitude feels continuous."""
    if tiles < 1:
        return data, lons

    half_span = tiles // 2
    offsets = range(-half_span, half_span + 1)
    wrapped_data = np.concatenate([data] * len(list(offsets)), axis=1)
    wrapped_lons = np.concatenate([
        lons + offset * period
        for offset in range(-half_span, half_span + 1)
    ])
    return wrapped_data, wrapped_lons


# ── Plot 1: 2-D Ocean Field Heatmap ──────────────────────────────────────────

def create_field_map(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,
    variable: str,
    title:    str,
) -> go.Figure:
    """
    Interactive lat/lon heatmap of a 2-D ocean field.

    Uses Plotly Heatmap with the variable-specific diverging or sequential
    colorscale. Hover labels include lat, lon and the field value with units.

    Parameters
    ----------
    data     : 2-D array [ny, nx].
    lats     : Latitude axis [ny].
    lons     : Longitude axis [nx].
    variable : Ocean field name (key in VARIABLE_META).
    title    : HTML-formatted figure title string.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    meta = VARIABLE_META.get(variable, VARIABLE_META["salt"])
    arr  = _mask_fill(data)
    vmin, vmax = _get_clim(arr, meta)
    label, units, cmap = meta["label"], meta["units"], meta["cmap"]
    wrapped_arr, wrapped_lons = _wrap_longitude_view(arr, lons)

    fig = go.Figure(
        go.Heatmap(
            z=wrapped_arr,
            x=wrapped_lons,
            y=lats,
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=dict(
                    text=f"{label}<br>({units})",
                    side="right",
                    font=dict(size=13),
                ),
                tickfont=dict(size=11),
                len=0.88,
                thickness=18,
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°<br>"
                "Lat: %{y:.2f}°<br>"
                f"{label}: %{{z:.4f}} {units}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, xanchor="center",
            font=dict(size=13, color="#e0e0e0"),
        ),
        xaxis=dict(
            title="Longitude (°)",
            showgrid=True, gridcolor="rgba(255,255,255,0.12)",
            tickformat=".1f", ticksuffix="°",
            color="#b0b0b0",
            range=[float(lons[0]), float(lons[-1])],
            fixedrange=False,
        ),
        yaxis=dict(
            title="Latitude (°)",
            showgrid=True, gridcolor="rgba(255,255,255,0.12)",
            tickformat=".1f", ticksuffix="°",
            color="#b0b0b0",
            # Keep equal aspect only when displaying the full globe
            scaleanchor="x" if (lons[-1] - lons[0]) > 270 else None,
            scaleratio=1.0,
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        margin=dict(l=70, r=20, t=90, b=65),
        height=520,
        dragmode="pan",
    )
    return fig


# ── Plot 2: Zonal Mean Profile ────────────────────────────────────────────────

def create_zonal_mean(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,   # noqa: ARG001 (kept for uniform signature)
    variable: str,
    title:    str,
) -> go.Figure:
    """
    Line chart of the longitude-averaged field vs latitude.

    Includes a shaded ±1-sigma band to show longitudinal spread.
    Useful for diagnosing meridional gradients and model biases.
    """
    meta = VARIABLE_META.get(variable, VARIABLE_META["salt"])
    arr  = _mask_fill(data)

    zmean  = np.nanmean(arr, axis=1)
    zstd   = np.nanstd(arr,  axis=1)

    fig = go.Figure()

    # Shaded ±σ band
    fig.add_trace(go.Scatter(
        x=np.concatenate([zmean + zstd, (zmean - zstd)[::-1]]),
        y=np.concatenate([lats,          lats[::-1]]),
        fill="toself",
        fillcolor="rgba(0,212,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ",
        showlegend=True,
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=zmean, y=lats,
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        name="Zonal mean",
    ))

    fig.update_layout(
        title=dict(
            text=f"Zonal Mean — {meta['label']}",
            x=0.5, xanchor="center", font=dict(size=12),
        ),
        xaxis=dict(
            title=f"{meta['label']} ({meta['units']})",
            color="#b0b0b0",
        ),
        yaxis=dict(
            title="Latitude (°)", ticksuffix="°",
            color="#b0b0b0",
        ),
        legend=dict(font=dict(size=11)),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        height=400,
        margin=dict(l=70, r=20, t=55, b=65),
    )
    return fig


# ── Plot 3: Value Histogram ───────────────────────────────────────────────────

def create_histogram(
    data:     np.ndarray,
    lats:     np.ndarray,   # noqa: ARG001
    lons:     np.ndarray,   # noqa: ARG001
    variable: str,
    title:    str,          # noqa: ARG001
) -> go.Figure:
    """
    Histogram of field values across the current geographic view.

    Annotates the mean and median for quick statistical reading.
    """
    meta  = VARIABLE_META.get(variable, VARIABLE_META["salt"])
    arr   = _mask_fill(data).flatten()
    valid = arr[np.isfinite(arr)]
    mean  = float(np.mean(valid))
    med   = float(np.median(valid))

    fig = go.Figure(go.Histogram(
        x=valid,
        nbinsx=80,
        marker_color="#00d4ff",
        opacity=0.75,
        name=variable,
    ))

    for val, label, color in [(mean, "Mean", "#ff6b6b"), (med, "Median", "#ffd93d")]:
        fig.add_vline(
            x=val, line_width=2, line_dash="dash", line_color=color,
            annotation_text=f"{label} {val:.3f}",
            annotation_font_color=color,
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(
            text=f"Distribution — {meta['label']}",
            x=0.5, xanchor="center", font=dict(size=12),
        ),
        xaxis=dict(
            title=f"{meta['label']} ({meta['units']})",
            color="#b0b0b0",
        ),
        yaxis=dict(title="Count", color="#b0b0b0"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        height=350,
        margin=dict(l=65, r=20, t=55, b=65),
    )
    return fig


# ── Plot 4: Temporal Snapshot Comparison (placeholder) ────────────────────────
#
# Example of a future multi-timestep comparison chart.
# Uncomment and flesh out to compare multiple time steps side by side.
#
# def create_time_comparison(data, lats, lons, variable, title):
#     ...

# ── PLOT REGISTRY ─────────────────────────────────────────────────────────────
# Maps string keys → callable plot factories with uniform signature:
#   fn(data, lats, lons, variable, title) -> go.Figure

PLOT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "field_map": {
        "fn":          create_field_map,
        "label":       "Ocean Field Map",
        "description": "2-D lat/lon heatmap with interactive colorbar",
    },
    "zonal_mean": {
        "fn":          create_zonal_mean,
        "label":       "Zonal Mean Profile",
        "description": "Longitude-averaged field vs latitude (±1σ band)",
    },
    "histogram": {
        "fn":          create_histogram,
        "label":       "Value Histogram",
        "description": "Distribution of field values in the current view",
    },
    # ── TO ADD A NEW PLOT ────────────────────────────────────────────────────
    # "my_plot": {
    #     "fn":          create_my_plot,   # same signature as the others
    #     "label":       "🔬 My New Plot",
    #     "description": "Short description shown in the sidebar tooltip",
    # },
}


def render_plot_box(
    key:      str,
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,
    variable: str,
    title:    str,
) -> None:
    """
    Central dispatcher: look up the plot factory and render it in Streamlit.

    Parameters
    ----------
    key      : Key in PLOT_REGISTRY.
    data     : 2-D field array.
    lats     : Latitude axis.
    lons     : Longitude axis.
    variable : Ocean field name.
    title    : Dynamic figure title.
    """
    entry = PLOT_REGISTRY.get(key)
    if entry is None:
        st.warning(f"Unknown plot key: '{key}'")
        return

    try:
        fig = entry["fn"](data, lats, lons, variable, title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.error(f"Rendering '{entry['label']}' failed.")
        st.code(traceback.format_exc())


# =============================================================================
# SECTION 6 — STREAMLIT UI COMPONENTS
# =============================================================================

def _configure_page() -> None:
    st.set_page_config(
        page_title="SciVis 2026 — Ocean Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        /* Global dark theme overrides */
        .stApp                   { background-color: #0e1117; color: #e0e0e0; }
        .block-container         { padding-top: 0.8rem; }
        h1, h2, h3               { color: #00d4ff; }
        .stSlider                { color: #e0e0e0; }
        div[data-testid="metric-container"] {
            background:    #1a1f2e;
            border:        1px solid rgba(0,212,255,0.25);
            border-radius: 8px;
            padding:       10px 14px;
        }
        div[data-testid="metric-container"] label {
            color: #7ec8e3 !important;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <h1 style='text-align:center;margin-bottom:0;color:#00d4ff;'>
            Atmospheric-Ocean Systems Visualization
        </h1>
        <h3 style='text-align:center;margin-top:4px;color:#7ec8e3;font-weight:400;'>
            IEEE SciVis Contest 2026 &nbsp;·&nbsp; NASA DYAMOND
            &nbsp;·&nbsp; LLC2160 Ocean Dataset
        </h3>
        <hr style='border:none;border-top:1px solid rgba(0,212,255,0.3);margin:8px 0 16px;'/>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, Any]:
    """
    Build all sidebar widgets and return the current parameter state.

    Returns
    -------
    dict with keys: variable, timestep, depth, lat_min, lat_max,
                    lon_min, lon_max, quality, quality_key,
                    show_zonal_mean, show_histogram.
    """
    with st.sidebar:
        st.markdown("## Controles")
        st.markdown("---")

        # ── Variable selector ─────────────────────────────────────────────
        st.markdown("### Variable")
        variable = st.selectbox(
            "Ocean Variable",
            options=list(VARIABLE_META.keys()),
            format_func=lambda k: f"{k} — {VARIABLE_META[k]['label']}",
            help=(
                "Select the ocean field to visualise. "
                "All variables are from the LLC2160 dataset. "
                "Additional variables can be added in DATASET_URLS."
            ),
        )
        st.markdown("---")

        # ── Time slider ───────────────────────────────────────────────────
        st.markdown("### Tiempo")
        timestep = st.slider(
            "Timestep",
            min_value=0,
            max_value=LLC2160_TOTAL_TIMESTEPS - 1,
            value=0,
            step=1,
            help=(
                f"Timestep 0 → {LLC2160_TOTAL_TIMESTEPS-1} "
                "(hourly from 2020-01-20). "
                "The heatmap updates automatically."
            ),
        )
        st.caption(f"≈ **{approx_datetime(timestep)}**")
        st.markdown("---")

        # ── Depth ─────────────────────────────────────────────────────────
        st.markdown("### Nivel de profundidad")
        depth = st.slider(
            "Depth Level",
            min_value=0,
            max_value=LLC2160_DEPTH_LEVELS - 1,
            value=0,
            step=1,
            help="0 = sea surface. The LLC2160 model has 90 vertical levels.",
        )
        depth_lbl = "Sea Surface" if depth == 0 else f"Level {depth} (≈ {depth * 50} m)"
        st.caption(f"Selected: **{depth_lbl}**")
        st.markdown("---")

        # ── Geographic region ─────────────────────────────────────────────
        st.markdown("### Región geográfica")

        preset_key = st.selectbox(
            "Quick Region",
            options=list(GEO_PRESETS.keys()),
            index=0,
            help="Preset regions override the manual lat/lon inputs below.",
        )

        preset = GEO_PRESETS.get(preset_key)
        if preset:
            default_lat_min, default_lat_max = preset[0], preset[1]
            default_lon_min, default_lon_max = preset[2], preset[3]
        else:
            default_lat_min, default_lat_max = -90.0, 90.0
            default_lon_min, default_lon_max = -180.0, 180.0

        col_a, col_b = st.columns(2)
        with col_a:
            lat_min = st.number_input("Lat Min (°)", value=float(default_lat_min),
                                      min_value=-90.0, max_value=89.0, step=1.0)
            lon_min = st.number_input("Lon Min (°)", value=float(default_lon_min),
                                      min_value=-180.0, max_value=179.0, step=1.0)
        with col_b:
            lat_max = st.number_input("Lat Max (°)", value=float(default_lat_max),
                                      min_value=-89.0, max_value=90.0, step=1.0)
            lon_max = st.number_input("Lon Max (°)", value=float(default_lon_max),
                                      min_value=-179.0, max_value=180.0, step=1.0)

        # Sanity checks
        if lat_min >= lat_max:
            st.error(" Lat Min must be less than Lat Max.")
            lat_min, lat_max = -90.0, 90.0
        if lon_min >= lon_max:
            st.error(" Lon Min must be less than Lon Max.")
            lon_min, lon_max = -180.0, 180.0

        st.markdown("---")

        # ── Resolution ────────────────────────────────────────────────────
        st.markdown("### Resolución de datos")
        quality_key = st.select_slider(
            "Data Quality",
            options=list(QUALITY_OPTIONS.keys()),
            value=DEFAULT_QUALITY_KEY,
            help=(
                "Controls the OpenVisus streaming resolution. "
                "'Medium' is recommended for initial exploration. "
                "Higher quality downloads more data and may be slower."
            ),
        )
        quality = QUALITY_OPTIONS[quality_key]
        st.markdown("---")

        # ── Additional plots ──────────────────────────────────────────────
        st.markdown("### Paneles adicionales")
        st.caption("Show extra visualisation panels below the main map.")
        show_zonal_mean = st.checkbox(
            "Zonal Mean Profile",
            value=False,
            help=PLOT_REGISTRY["zonal_mean"]["description"],
        )
        show_histogram = st.checkbox(
            "Value Histogram",
            value=False,
            help=PLOT_REGISTRY["histogram"]["description"],
        )
        # ── TO ADD A NEW PANEL ─────────────────────────────────────────
        # show_my_plot = st.checkbox("My New Plot", value=False,
        #     help=PLOT_REGISTRY["my_plot"]["description"])

        st.markdown("---")
        st.markdown("### 💾 Cache local")
        try:
            _files = [f for f in os.listdir(_SLICE_DIR) if f.endswith(".npz")]
            _n  = len(_files)
            _mb = sum(os.path.getsize(os.path.join(_SLICE_DIR, f)) for f in _files) / 1e6
            if _n == 0:
                st.caption("Sin datos locales. Ejecuta `python download_data.py` para carga instantanea.")
            else:
                st.caption(f"OK: {_n} slices en disco ({_mb:.1f} MB)")
            if _n > 0 and st.button("Limpiar cache"):
                for _f in _files:
                    try: os.remove(os.path.join(_SLICE_DIR, _f))
                    except OSError: pass
                st.cache_data.clear()
                st.rerun()
        except Exception:
            pass
        st.markdown("---")
        st.caption(
            "**Data:** NASA DYAMOND LLC2160  \n"
            "**Access:** OpenVisus / NSDF  \n"
            "**Contest:** [SciVis 2026](https://sciviscontest2026.github.io)"
        )

        return {
            "variable":        variable,
            "timestep":        int(timestep),
            "depth":           int(depth),
            "lat_min":         float(lat_min),
            "lat_max":         float(lat_max),
            "lon_min":         float(lon_min),
            "lon_max":         float(lon_max),
            "quality":         int(quality),
            "quality_key":     quality_key,
            "show_zonal_mean": show_zonal_mean,
            "show_histogram":  show_histogram,
            # Extend here for additional panel flags
        }


def render_status_banner(using_demo: bool, params: Dict[str, Any]) -> None:
    """Show a data-source banner (live data vs demo mode)."""
    if using_demo:
        st.info(
            " **Demo mode** — displaying synthetic data.  "
            "Install `OpenVisus` (`pip install OpenVisus`) and ensure "
            "network access to the NSDF server for real NASA data."
        )
    else:
        st.success(
            f" **Live data** via OpenVisus / NSDF  ·  "
            f"Variable: **{params['variable']}**  ·  "
            f"Quality: **{params['quality_key']}**  ·  "
            f"Timestep: **{params['timestep']}** "
            f"({approx_datetime(params['timestep'])})"
        )


def render_metrics(data: Optional[np.ndarray], params: Dict[str, Any]) -> None:
    """Five-column statistics bar above the main chart."""
    meta   = VARIABLE_META.get(params["variable"], {})
    units  = meta.get("units", "")
    shape  = data.shape if data is not None else (0, 0)
    valid  = data[np.isfinite(data)] if data is not None else np.array([])
    mb     = data.nbytes / 1e6 if data is not None else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variable",       params["variable"])
    c2.metric("Timestep",       f"{params['timestep']:,}")
    c3.metric("Grid (px)",      f"{shape[1]} × {shape[0]}" if shape[0] else "—")
    c4.metric(f"Min ({units})", f"{float(valid.min()):.3f}" if valid.size else "—")
    c5.metric(f"Max ({units})", f"{float(valid.max()):.3f}" if valid.size else "—")

    if mb > 20:
        st.warning(
            f"⚠️ Array is **{mb:.0f} MB** — consider using a lower quality setting "
            "or a smaller geographic region to avoid browser memory issues."
        )


def build_title(params: Dict[str, Any]) -> str:
    """Compose the dynamic chart title from current parameters."""
    meta  = VARIABLE_META.get(params["variable"], {})
    label = meta.get("label", params["variable"])
    units = meta.get("units", "")
    depth_str = "Surface" if params["depth"] == 0 else f"Depth Lv {params['depth']}"
    geo_str = (
        f"Lat [{params['lat_min']:.1f}°, {params['lat_max']:.1f}°]  "
        f"Lon [{params['lon_min']:.1f}°, {params['lon_max']:.1f}°]"
    )
    return (
        f"<b>{label} ({units})</b>  ·  "
        f"Timestep {params['timestep']}  ·  {depth_str}<br>"
        f"<sup>{geo_str}</sup>"
    )


# =============================================================================
# SECTION 7 — MAIN APPLICATION
# =============================================================================

def main() -> None:
    _configure_page()
    render_header()

    # ── 1. Sidebar controls ────────────────────────────────────────────────
    params = render_sidebar()

    # ── 2. Compute normalised coordinate ranges for OpenVisus ──────────────
    x_lo, y_lo = latlon_to_norm(params["lat_min"], params["lon_min"])
    x_hi, y_hi = latlon_to_norm(params["lat_max"], params["lon_max"])
    x_range: Tuple[float, float] = (x_lo, x_hi)
    y_range: Tuple[float, float] = (y_lo, y_hi)

    # ── 3. Fetch data ──────────────────────────────────────────────────────
    # Verificar si el dato esta en disco antes de mostrar spinner
    _key_fname = _slice_filename(
        params["variable"], params["timestep"],
        params["depth"], params["quality"]
    )
    _is_cached = os.path.exists(os.path.join(_SLICE_DIR, _key_fname))
    _spinner_msg = (
        "⚡ Cargando desde disco local…"
        if _is_cached else
        "📡 Descargando desde NSDF (ejecuta download_data.py para pre-cargar)…"
    )
    with st.spinner(_spinner_msg):
        data: Optional[np.ndarray] = None
        using_demo = False
        _conn_error: Optional[str] = None

        if not OPENVISUS_AVAILABLE:
            _conn_error = "OpenVisus not installed — run: `pip install OpenVisus`"
        else:
            # Test the dataset connection first so we can report a clear error
            _db = _connect_dataset(params["variable"])
            if _db is None:
                _conn_error = (
                    f"Could not connect to NSDF server for variable "
                    f"**{params['variable']}**. "
                    "Check your internet connection and that the NSDF server "
                    "`nsdf-climate3-origin.nationalresearchplatform.org:50098` "
                    "is reachable."
                )
            else:
                data = fetch_ocean_slice(
                    variable  = params["variable"],
                    timestep  = params["timestep"],
                    depth     = params["depth"],
                    quality   = params["quality"],
                    x_range   = x_range,
                    y_range   = y_range,
                )
                if data is None:
                    _conn_error = (
                        f"Connected to dataset but `db.read()` returned no data "
                        f"(var={params['variable']}, t={params['timestep']}, "
                        f"depth={params['depth']}). "
                        "Try a lower quality setting or timestep 0."
                    )

        if data is None:
            using_demo = True
            if _conn_error:
                st.warning(f"⚠️ {_conn_error}  \n↳ Falling back to **demo/synthetic data**.")
            data = _generate_demo_slice(
                variable  = params["variable"],
                timestep  = params["timestep"],
                lat_range = (params["lat_min"], params["lat_max"]),
                lon_range = (params["lon_min"], params["lon_max"]),
            )

    # ── 4. Build coordinate axes matching returned array shape ─────────────
    ny, nx = data.shape
    lats, lons = make_axes(
        params["lat_min"], params["lat_max"],
        params["lon_min"], params["lon_max"],
        ny, nx,
    )

    # ── 5. Dynamic title ───────────────────────────────────────────────────
    title = build_title(params)

    # ── 6. Status banner + statistics ─────────────────────────────────────
    render_status_banner(using_demo, params)
    render_metrics(data, params)
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,212,255,0.2);margin:8px 0;'/>",
        unsafe_allow_html=True,
    )

    # ── 7. Primary visualisation (always shown) ────────────────────────────
    render_plot_box(
        key      = "field_map",
        data     = data,
        lats     = lats,
        lons     = lons,
        variable = params["variable"],
        title    = title,
    )

    # ── 8. Optional secondary panels (user-toggled in sidebar) ────────────
    secondary_panels: List[Tuple[bool, str]] = [
        (params["show_zonal_mean"], "zonal_mean"),
        (params["show_histogram"],  "histogram"),
        # ── TO ADD A NEW PANEL ─────────────────────────────────────────
        # (params["show_my_plot"], "my_plot"),
    ]

    active = [(show, key) for show, key in secondary_panels if show]
    if active:
        # Two panels per row
        for i in range(0, len(active), 2):
            cols = st.columns(min(2, len(active) - i))
            for j, (_, key) in enumerate(active[i:i+2]):
                with cols[j]:
                    render_plot_box(
                        key      = key,
                        data     = data,
                        lats     = lats,
                        lons     = lons,
                        variable = params["variable"],
                        title    = title,
                    )

    # ── 9. Footer ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr style='border:none;border-top:1px solid rgba(0,212,255,0.15);margin:16px 0 6px;'/>
        <p style='text-align:center;color:#506070;font-size:0.78rem;'>
            NASA DYAMOND LLC2160 Ocean Simulation &nbsp;·&nbsp;
            OpenVisus / NSDF / Seal Storage &nbsp;·&nbsp;
            IEEE SciVis Contest 2026
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
