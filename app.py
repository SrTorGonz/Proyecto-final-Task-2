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

try:
    from global_land_mask import globe  # pip install global-land-mask
    GLOBAL_LAND_MASK_AVAILABLE = True
except ImportError:
    GLOBAL_LAND_MASK_AVAILABLE = False

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

DATASET_FIELDS: Dict[str, str] = {
    "Theta": "theta",
}

# Variable display metadata — label, units, default colour range, Plotly colorscale
VARIABLE_META: Dict[str, Dict[str, Any]] = {
    "salt":  {"label": "Sea Surface Salinity",  "units": "g kg⁻¹", "cmap": "haline",  "vmin": 31.0,  "vmax": 38.0},
    "Theta": {"label": "Potential Temperature", "units": "°C",     "cmap": "thermal", "vmin": -2.0,  "vmax": 32.0},
    "u":     {"label": "Zonal Velocity (ARCO)", "units": "m/s",    "cmap": "balance", "vmin": -1.5,  "vmax": 1.5},
    "v":     {"label": "Meridional Velocity",   "units": "m/s",    "cmap": "balance", "vmin": -1.5,  "vmax": 1.5},
    "w":     {"label": "Vertical Velocity",     "units": "m/s",    "cmap": "delta",   "vmin": -0.01, "vmax": 0.01},
    "GEOS_U": {"label": "GEOS Eastward Wind (U)",  "units": "m/s",  "cmap": "balance", "vmin": -40.0, "vmax": 40.0},
    "GEOS_V": {"label": "GEOS Northward Wind (V)", "units": "m/s",  "cmap": "balance", "vmin": -40.0, "vmax": 40.0},
    "GEOS_P": {"label": "GEOS Mid-level Pressure (P)", "units": "hPa", "cmap": "turbo",   "vmin": 50.0,  "vmax": 1050.0},
    "GEOS_T": {"label": "GEOS Air Temperature (T)",    "units": "K",   "cmap": "thermal", "vmin": 180.0, "vmax": 310.0},
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

# Maximum number of points used in the temperature time-series panel.
TEMP_SERIES_MAX_POINTS: int = 36

# Temperature history pulls from cached Theta slices when available.
TEMP_HISTORY_VARIABLE: str = "Theta"

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
def _connect_dataset(variable: str, face: int = 0) -> Optional[Any]:
    """
    Open a persistent OpenVisus connection for a given variable.

    Cached as a Streamlit resource.

    Parameters
    ----------
    variable : str
        Field name.
    face : int
        GEOS face index (0-5), ignored for LLC2160 ocean variables.

    Returns
    -------
    OpenVisus PyDataset object or None.
    """
    if not OPENVISUS_AVAILABLE:
        return None
    
    if variable.startswith("GEOS_"):
        var_char = variable.split("_")[1].lower()  # 'u', 'v', 'p', 't'
        url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{var_char.upper()}/{var_char}_face_{face}_depth_52_time_0_10269.idx"
    else:
        url = DATASET_URLS.get(variable)
        
    if not url:
        return None
    try:
        return ov.LoadDataset(url)
    except Exception as exc:
        logging.warning("OpenVisus connection failed for '%s' (face %d): %s", variable, face, exc)
        return None


def _dataset_field_name(variable: str, db: Optional[Any] = None, face: int = 0) -> str:
    """Return the OpenVisus field name for a display variable key."""
    if variable in DATASET_FIELDS:
        return DATASET_FIELDS[variable]
    if db is None:
        db = _connect_dataset(variable, face)
    try:
        return db.getField().name if db is not None else variable
    except Exception:
        return variable



# =============================================================================
# SECCION 2b — DISCO-CACHE
# Generado por download_data.py. La clave es simple (no usa hash de floats).
# =============================================================================

def _slice_filename(variable: str, timestep: int, depth: int, quality: int, face: int = 0) -> str:
    """Nombre de archivo — DEBE coincidir con download_data.py."""
    if variable.startswith("GEOS_"):
        return f"{variable}_face{face}_t{timestep:05d}_d{depth}_q{quality}_global.npz"
    return f"{variable}_t{timestep:05d}_d{depth}_q{quality}_global.npz"


def _disk_load(variable: str, timestep: int, depth: int,
               quality: int, face: int = 0) -> Optional[np.ndarray]:
    """Carga un slice global del disco. None si no existe."""
    path = os.path.join(_SLICE_DIR, _slice_filename(variable, timestep, depth, quality, face))
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
               quality: int, arr: np.ndarray, face: int = 0) -> None:
    """Guarda un slice en disco (escritura atomica)."""
    path = os.path.join(_SLICE_DIR, _slice_filename(variable, timestep, depth, quality, face))
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
    face:      int = 0,
) -> Optional[np.ndarray]:
    """
    Download a 2-D horizontal slice from the dataset.
    """
    # ── Capa 1: disco-cache 
    _global = _disk_load(variable, timestep, depth, quality, face)
    if _global is not None:
        arr = _crop_to_region(_global, x_range, y_range)
        ny, nx = arr.shape
        sy = max(1, ny // MAX_RENDER_PX)
        sx = max(1, nx // MAX_RENDER_PX)
        if sy > 1 or sx > 1:
            arr = arr[::sy, ::sx]
        return arr

    # ── Capa 2: descarga en tiempo real
    db = _connect_dataset(variable, face)
    if db is None:
        return None

    try:
        field_name = _dataset_field_name(variable, db, face)
        raw = db.read(
            time=timestep,
            field=field_name,
            quality=quality,
            z=[depth, depth + 1],
        )

        if raw is None:
            return None

        arr = np.array(raw, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr[0]

        _disk_save(variable, timestep, depth, quality, arr, face)

        arr = _crop_to_region(arr, x_range, y_range)
        ny, nx = arr.shape
        stride_y = max(1, ny // MAX_RENDER_PX)
        stride_x = max(1, nx // MAX_RENDER_PX)
        if stride_y > 1 or stride_x > 1:
            arr = arr[::stride_y, ::stride_x]

        return arr

    except Exception as exc:
        logging.warning("read failed (var=%s, face=%d, t=%d): %s", variable, face, timestep, exc)
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

    elif variable == "GEOS_U":
        # Eastward wind: trade winds negative near equator, westerlies positive mid-lat
        data = (
            -8.0 * np.cos(np.radians(LAT * 3.0))
            + 12.0 * np.sin(np.radians(LAT * 2.0))
            + 4.0  * np.sin(np.radians(LON * 0.5) + phase)
            + 3.0  * np.cos(np.radians(LON + LAT) + phase)
            + 1.5  * np.random.randn(n, n)
        )

    elif variable == "GEOS_V":
        # Northward wind: meridional Hadley/Rossby component
        data = (
            3.0  * np.sin(np.radians(LON * 1.0) + phase)
            + 4.0  * np.cos(np.radians(LON * 0.7 + LAT))
            - 2.0  * np.sin(np.radians(LAT * 4.0))
            + 1.0  * np.random.randn(n, n)
        )

    elif variable == "u":
        # Ocean zonal current: gyre structure
        data = (
            0.6  * np.cos(np.radians(LAT * 2.5))
            + 0.25 * np.sin(np.radians(LON * 0.8) + phase)
            + 0.12 * np.cos(np.radians(LON + LAT * 1.5))
            + 0.05 * np.random.randn(n, n)
        )

    elif variable == "v":
        # Ocean meridional current: boundary current structure
        data = (
            0.25 * np.sin(np.radians(LON * 1.0) + phase)
            + 0.18 * np.cos(np.radians(LON * 0.5 + LAT * 2.0))
            + 0.04 * np.random.randn(n, n)
        )

    else:
        # Generic diverging field (pressure, etc.)
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
    
    is_geos = variable.startswith("GEOS_")
    
    if is_geos:
        wrapped_arr = arr
        wrapped_lons = lons
        x_title = "X (Face Relative)"
        y_title = "Y (Face Relative)"
        x_tick_suffix = ""
        y_tick_suffix = ""
        x_tick_format = "d"
        y_tick_format = "d"
        hover_xy = "X: %{x}<br>Y: %{y}<br>"
    else:
        wrapped_arr, wrapped_lons = _wrap_longitude_view(arr, lons)
        x_title = "Longitude (°)"
        y_title = "Latitude (°)"
        x_tick_suffix = "°"
        y_tick_suffix = "°"
        x_tick_format = ".1f"
        y_tick_format = ".1f"
        hover_xy = "Lon: %{x:.2f}°<br>Lat: %{y:.2f}°<br>"

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
                hover_xy +
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
            title=x_title,
            showgrid=True, gridcolor="rgba(255,255,255,0.12)",
            tickformat=x_tick_format, ticksuffix=x_tick_suffix,
            color="#b0b0b0",
            range=[float(lons[0]), float(lons[-1])],
            fixedrange=False,
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor="rgba(255,255,255,0.12)",
            tickformat=y_tick_format, ticksuffix=y_tick_suffix,
            color="#b0b0b0",
            # Keep equal aspect only when displaying the full globe
            scaleanchor="x" if (not is_geos and (lons[-1] - lons[0]) > 270) else None,
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


# ── Plot 5: Hovmöller Diagram ─────────────────────────────────────────────────

def create_hovmoller(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,   # noqa: ARG001
    variable: str,
    title:    str,          # noqa: ARG001
) -> go.Figure:
    """Hovmöller diagram: heatmap latitud × tiempo con la media zonal de la variable seleccionada."""
    meta     = VARIABLE_META.get(variable, VARIABLE_META["salt"])
    lat_step = 4

    timesteps_to_try = list(range(0, 10))
    frames: List[Optional[np.ndarray]] = []

    for t in timesteps_to_try:
        arr = _disk_load_any_quality(variable, t, 0, face=0)
        if arr is None:
            arr = _generate_demo_slice(
                variable, t,
                lat_range=(lats[0] if len(lats) > 0 else -90, lats[-1] if len(lats) > 0 else 90),
                lon_range=(-180, 180),
            )
        arr_sub    = arr[::lat_step, :]
        zonal_mean = np.nanmean(_mask_fill(arr_sub), axis=1)
        frames.append(zonal_mean)

    ref_len = max((len(f) for f in frames if f is not None), default=1)
    frames  = [f if f is not None else np.full(ref_len, np.nan) for f in frames]

    matrix      = np.column_stack(frames)
    lat_indices = np.arange(matrix.shape[0]) * lat_step - 90
    time_labels = [f"t={t}" for t in timesteps_to_try]
    vmin, vmax  = _get_clim(matrix, meta)

    fig = go.Figure(go.Heatmap(
        z           = matrix,
        x           = time_labels,
        y           = lat_indices,
        colorscale  = meta["cmap"],
        zmin        = vmin,
        zmax        = vmax,
        colorbar    = dict(title=f"{meta['label']} ({meta['units']})", tickfont=dict(size=11)),
        hoverongaps = False,
        hovertemplate=f"Timestep: %{{x}}<br>Latitud aprox: %{{y:.1f}}°<br>{meta['label']}: %{{z:.3f}} {meta['units']}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Hovmöller — {meta['label']}</b><br><sup>Media zonal por latitud × timestep</sup>",
            x=0.5, xanchor="center", font=dict(size=14, color="#dce8f0"),
        ),
        xaxis=dict(title="Timestep", tickangle=-45, color="#b0b0b0"),
        yaxis=dict(title="Latitud aproximada (°)", ticksuffix="°", autorange="reversed", color="#b0b0b0"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        height=480,
        margin=dict(l=70, r=40, t=80, b=60),
    )
    return fig

# ── Plot 6: Bubble Map — Wind Stress y Anomalía SST ──────────────────────────

def create_bubble_windstress(
    data:     np.ndarray,
    lats:     np.ndarray,
    lons:     np.ndarray,
    variable: str,
    title:    str,  # noqa: ARG001
) -> go.Figure:
    """
    Mapa de burbujas geoespacial: tamaño = wind stress τ = ρ·Cd·|V|²
    color = anomalía de SST (Theta - media global).
    Muestra dónde el forzamiento atmosférico produce respuesta térmica oceánica.
    """
    import plotly.express as px

    # ── Constantes físicas ────────────────────────────────────────────────
    RHO_AIR = 1.225   # densidad del aire kg/m³
    CD      = 1.3e-3  # coeficiente de arrastre adimensional (típico oceánico)

    # ── Cargar vientos GEOS_U y GEOS_V desde disco ────────────────────────
    wind_u = _disk_load_any_quality("GEOS_U", 0, 0, face=0)
    wind_v = _disk_load_any_quality("GEOS_V", 0, 0, face=0)

    # ── Cargar SST (Theta) desde disco ────────────────────────────────────
    sst = _disk_load_any_quality("Theta", 0, 0, face=0)

    # ── Fallback a datos sintéticos si no hay cache ───────────────────────
    lat_range = (float(lats[0]), float(lats[-1])) if len(lats) > 1 else (-90.0, 90.0)
    lon_range = (float(lons[0]), float(lons[-1])) if len(lons) > 1 else (-180.0, 180.0)

    if wind_u is None:
        wind_u = _generate_demo_slice("GEOS_U", 0, lat_range, lon_range)
    if wind_v is None:
        wind_v = _generate_demo_slice("GEOS_V", 0, lat_range, lon_range)
    if sst is None:
        sst = _generate_demo_slice("Theta", 0, lat_range, lon_range)

   # ── Alinear dimensiones con resize al shape del viento ────────────────
    target_h, target_w = wind_u.shape
    def _resize(arr, h, w):
        ri = np.linspace(0, arr.shape[0]-1, h, dtype=int)
        ci = np.linspace(0, arr.shape[1]-1, w, dtype=int)
        return arr[np.ix_(ri, ci)]

    wind_v = _resize(wind_v, target_h, target_w)
    sst    = _resize(sst,    target_h, target_w)

    # ── Submuestreo para que el mapa sea legible (≈ 600 burbujas) ─────────
    step = max(1, target_h // 35)
    wind_u_s = wind_u[::step, ::step]
    wind_v_s = wind_v[::step, ::step]
    sst_s    = sst[::step, ::step]

    ny, nx = wind_u_s.shape
    lat_1d = np.linspace(lat_range[0], lat_range[1], ny)
    lon_1d = np.linspace(lon_range[0], lon_range[1], nx)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # ── Calcular wind stress τ = ρ · Cd · (U² + V²) ──────────────────────
    speed_sq   = _mask_fill(wind_u_s)**2 + _mask_fill(wind_v_s)**2
    tau        = RHO_AIR * CD * speed_sq          # N/m²

    # ── Calcular anomalía de SST ──────────────────────────────────────────
    sst_clean  = _mask_fill(sst_s)
    sst_mean   = float(np.nanmean(sst_clean))
    sst_anom   = sst_clean - sst_mean              # °C respecto a la media global

    # ── Aplanar y filtrar NaN ─────────────────────────────────────────────
    lat_flat  = lat_grid.ravel()
    lon_flat  = lon_grid.ravel()
    tau_flat  = tau.ravel()
    anom_flat = sst_anom.ravel()

    mask = np.isfinite(tau_flat) & np.isfinite(anom_flat) & (tau_flat > 0)
    lat_flat  = lat_flat[mask]
    lon_flat  = lon_flat[mask]
    tau_flat  = tau_flat[mask]
    anom_flat = anom_flat[mask]

    # ── Filtrar puntos sobre tierra ───────────────────────────────────────
    if GLOBAL_LAND_MASK_AVAILABLE:
        is_ocean = ~globe.is_land(lat_flat, lon_flat)
        lat_flat  = lat_flat[is_ocean]
        lon_flat  = lon_flat[is_ocean]
        tau_flat  = tau_flat[is_ocean]
        anom_flat = anom_flat[is_ocean]

    # ── Normalizar tamaño de burbuja (0–40 px) ────────────────────────────
    tau_norm = (tau_flat - tau_flat.min()) / (tau_flat.max() - tau_flat.min() + 1e-9)
    size_px  = 4 + tau_norm * 36   # min 4, max 40

    fig = px.scatter_geo(
        lat        = lat_flat,
        lon        = lon_flat,
        size       = size_px,
        color      = anom_flat,
        color_continuous_scale = "RdBu_r",
        range_color= [-5, 5],
        size_max   = 12,
        opacity    = 0.65,
        projection = "natural earth",
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=0),
        ),
        hovertemplate=(
            "Lat: %{lat:.1f}°  Lon: %{lon:.1f}°<br>"
            "Anomalía SST: %{marker.color:.2f} °C<br>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Wind Stress → SST: Transferencia de Momentum Atmósfera–Océano</b><br>"
                "<sup>Tamaño = τ (ρ·Cd·|V|²)  ·  Color = Anomalía de Temperatura Superficial del Mar</sup>"
            ),
            x=0.5, xanchor="center", xref="paper",
            font=dict(size=14, color="#dce8f0"),
        ),
        geo=dict(
            showland        = True,
            landcolor       = "#1e2a3a",
            showocean       = True,
            oceancolor      = "#0a1628",
            showcoastlines  = True,
            coastlinecolor  = "#4a6fa5",
            coastlinewidth  = 0.8,
            showlakes       = True,
            lakecolor       = "#0a1628",
            showrivers      = False,
            showframe       = False,
            showcountries   = True,
            countrycolor    = "#2a3f55",
            countrywidth    = 0.4,
            projection_type = "natural earth",
            bgcolor         = "#0e1117",
            lataxis         = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            lonaxis         = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        ),
        coloraxis_colorbar=dict(
            title    = "Anomalía SST (°C)",
            tickfont = dict(size=11, color="#e0e0e0"),
            len      = 0.6,
            thickness= 14,
            x        = 1.01,
        ),
        paper_bgcolor = "#0e1117",
        font          = dict(color="#e0e0e0"),
        height        = 560,
        margin        = dict(l=0, r=80, t=80, b=10),
    )
    return fig

# ── PLOT REGISTRY ─────────────────────────────────────────────────────────────
# Maps string keys → callable plot factories with uniform signature:
#   fn(data, lats, lons, variable, title) -> go.Figure

def _sample_vector_field(
    east: np.ndarray,
    north: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    stride: int = 7,
    polar_stride: int = 12,
    ocean_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse lon/lat/vector arrays suitable for a readable quiver layer."""
    east_arr = _mask_fill(east)
    north_arr = _mask_fill(north)
    rows, cols = east_arr.shape

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    row_grid, col_grid = np.meshgrid(
        np.arange(rows),
        np.arange(cols),
        indexing="ij",
    )

    valid = np.isfinite(east_arr) & np.isfinite(north_arr)

    if ocean_mask is not None:
        valid &= ocean_mask

    standard_keep = (row_grid % stride == 0) & (col_grid % stride == 0)
    polar_keep = (row_grid % polar_stride == 0) & (col_grid % polar_stride == 0)
    valid &= np.where(np.abs(lat_grid) > 55.0, polar_keep, standard_keep)

    return (
        lon_grid[valid].ravel(),
        lat_grid[valid].ravel(),
        east_arr[valid].ravel(),
        north_arr[valid].ravel(),
    )


def _vector_arrow_traces(
    lon: np.ndarray,
    lat: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    *,
    name: str,
    color: str,
    arrow_length_degrees: float,
    line_color: str,
    ocean_only: bool = False,
    line_width: float = 1.5,
) -> Tuple[go.Scattergeo, go.Scattergeo, float]:
    """Draw geographic vectors as shaft segments plus arrowhead markers."""
    magnitude = np.sqrt(east ** 2 + north ** 2)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(magnitude) & (magnitude > 0)
    valid &= np.abs(lat) <= 80.0
    if ocean_only and GLOBAL_LAND_MASK_AVAILABLE:
        valid &= ~globe.is_land(lat, lon)

    if not np.any(valid):
        empty_lines = go.Scattergeo(
            lon=[],
            lat=[],
            mode="lines",
            line=dict(color=line_color, width=line_width),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
        empty_heads = go.Scattergeo(
            lon=[],
            lat=[],
            mode="markers",
            name=name,
            showlegend=True,
            hoverinfo="skip",
        )
        return empty_lines, empty_heads, 1.0

    lon = lon[valid]
    lat = lat[valid]
    east = east[valid]
    north = north[valid]
    magnitude = magnitude[valid]

    reference = float(np.nanpercentile(magnitude, 95))
    if not np.isfinite(reference) or reference <= 0:
        reference = 1.0

    # Angle clockwise from North — correct bearing for Plotly angleref="up"
    angle = np.degrees(np.arctan2(east, north))

    # Log-scale visual length: makes slow ocean currents AND fast winds both legible
    visual_ratio = np.log1p(magnitude) / np.log1p(reference)
    visual_ratio = np.clip(visual_ratio, 0.15, 1.0)  # min 15% so tiny vectors still show
    length_degrees = arrow_length_degrees * visual_ratio

    angle_rad = np.radians(angle)
    lon_tip = lon + np.sin(angle_rad) * length_degrees
    lat_tip = lat + np.cos(angle_rad) * length_degrees

    # Arrowhead size: SMALL and fixed — 7px base, 10px max.
    # Variable size was causing the bloated triangle look.
    head_size = (5.0 + 5.0 * visual_ratio).clip(5.0, 10.0)

    in_bounds = (lon_tip >= -180.0) & (lon_tip <= 180.0) & (lat_tip >= -80.0) & (lat_tip <= 80.0)
    if ocean_only and GLOBAL_LAND_MASK_AVAILABLE:
        tip_is_ocean = np.zeros_like(in_bounds, dtype=bool)
        tip_is_ocean[in_bounds] = ~globe.is_land(lat_tip[in_bounds], lon_tip[in_bounds])
        in_bounds &= tip_is_ocean
    lon = lon[in_bounds]
    lat = lat[in_bounds]
    lon_tip = lon_tip[in_bounds]
    lat_tip = lat_tip[in_bounds]
    angle = angle[in_bounds]
    head_size = head_size[in_bounds]

    # Build shaft segments (tail → tip, separated by None for gap between arrows)
    lons_seg: List[Optional[float]] = []
    lats_seg: List[Optional[float]] = []
    for lo, la, lo2, la2 in zip(lon, lat, lon_tip, lat_tip):
        lons_seg.extend([float(lo), float(lo2), None])
        lats_seg.extend([float(la), float(la2), None])

    line_trace = go.Scattergeo(
        lon=lons_seg,
        lat=lats_seg,
        mode="lines",
        line=dict(color=line_color, width=1.2),
        name=f"{name} shafts",
        showlegend=False,
        hoverinfo="skip",
    )
    head_trace = go.Scattergeo(
        lon=lon_tip,
        lat=lat_tip,
        mode="markers",
        marker=dict(
            symbol="arrow",
            size=head_size,
            color=color,
            angleref="up",
            angle=angle,
            line=dict(width=0),   # no border on arrowhead — cleaner look
        ),
        name=f"{name} · P95={reference:.1f}m/s",
        showlegend=True,
        hoverinfo="skip",
    )
    return line_trace, head_trace, reference


def _resize_nearest(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array with nearest-neighbour indexing for vector overlays."""
    rows, cols = arr.shape
    target_rows, target_cols = shape
    row_idx = np.linspace(0, rows - 1, target_rows, dtype=int)
    col_idx = np.linspace(0, cols - 1, target_cols, dtype=int)
    return arr[np.ix_(row_idx, col_idx)]


def _build_ocean_mask(mask_source: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Create a simple ocean-validity mask from Theta/salt-style data."""
    source = _resize_nearest(_mask_fill(mask_source), shape)
    return np.isfinite(source) & (np.abs(source) > 1e-12)


def create_wind_current_quiver(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    ocean_mask_source: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    params: Dict[str, Any],
) -> go.Figure:
    """
    Quiver map of GEOS winds overlaid with LLC2160 ocean currents.

    Wind arrows are blue and current arrows are orange, making wind-driven
    circulation and Ekman transport easier to inspect directly.
    """
    # ── Determine target grid resolution ──────────────────────────────────────
    # Cap at 128×256 for a global view — enough detail without OOM.
    # For regional views (small domain) use the natural data resolution.
    lat_span = params["lat_max"] - params["lat_min"]
    lon_span = (params["lon_max"] - params["lon_min"]) % 360 or 360
    is_global = lat_span > 140 and lon_span > 330
    max_rows = 128 if is_global else 160
    max_cols = 256 if is_global else 320

    target_shape = (
        min(wind_u.shape[0], wind_v.shape[0], current_u.shape[0], current_v.shape[0], ocean_mask_source.shape[0]),
        min(wind_u.shape[1], wind_v.shape[1], current_u.shape[1], current_v.shape[1], ocean_mask_source.shape[1]),
    )
    target_shape = (min(target_shape[0], max_rows), min(target_shape[1], max_cols))

    wind_u = _resize_nearest(wind_u, target_shape)
    wind_v = _resize_nearest(wind_v, target_shape)
    current_u = _resize_nearest(current_u, target_shape)
    current_v = _resize_nearest(current_v, target_shape)
    ocean_mask = _build_ocean_mask(ocean_mask_source, target_shape)
    lats, lons = make_axes(
        params["lat_min"], params["lat_max"],
        params["lon_min"], params["lon_max"],
        target_shape[0], target_shape[1],
    )

    # ── Stride: fewer points = cleaner arrows, no overlap ─────────────────────
    # Global: stride=8 → ~16×32 ≈ 512 wind arrows — readable, not crowded.
    # Regional: stride=5 for more spatial detail.
    base_stride = 8 if is_global else 5
    polar_stride = 14 if is_global else 9

    wx, wy, wu, wv = _sample_vector_field(
        wind_u,
        wind_v,
        lats,
        lons,
        stride=base_stride,
        polar_stride=polar_stride,
    )
    cx, cy, cu, cv = _sample_vector_field(
        current_u,
        current_v,
        lats,
        lons,
        stride=base_stride + 1,
        polar_stride=polar_stride,
        ocean_mask=ocean_mask,
    )

    # ── Arrow length in geographic degrees ────────────────────────────────────
    # Rule of thumb: one arrow should span roughly one grid cell width so
    # adjacent arrows don't overlap.  Grid cell ≈ lon_span / num_cols.
    grid_cell_deg = lon_span / target_shape[1]
    # Wind arrows: slightly longer than one cell; currents: slightly shorter.
    wind_arrow_deg  = max(4.0, grid_cell_deg * base_stride * 0.85)
    ocean_arrow_deg = max(3.5, grid_cell_deg * (base_stride + 1) * 0.75)

    wind_lines, wind_heads, wind_ref = _vector_arrow_traces(
        wx,
        wy,
        wu,
        wv,
        name="Vientos GEOS (U,V)",
        color="#4da6ff",
        line_color="rgba(77,166,255,0.55)",
        arrow_length_degrees=wind_arrow_deg,
    )
    current_lines, current_heads, current_ref = _vector_arrow_traces(
        cx,
        cy,
        cu,
        cv,
        name="Corrientes LLC2160 (u,v)",
        color="#ff8c42",
        line_color="rgba(255,140,66,0.6)",
        arrow_length_degrees=ocean_arrow_deg,
        ocean_only=True,
    )

    fig = go.Figure(data=[wind_lines, current_lines, wind_heads, current_heads])

    fig.update_geos(
        projection_type="equirectangular",
        lonaxis=dict(range=[-180, 180], showgrid=True,
                     gridcolor="rgba(255,255,255,0.07)", dtick=30),
        lataxis=dict(range=[-80, 80],  showgrid=True,
                     gridcolor="rgba(255,255,255,0.07)", dtick=30),
        showland=True,    landcolor="#252836",   # distinct grey-blue land
        showocean=True,   oceancolor="#09152a",  # deep navy ocean
        showlakes=True,   lakecolor="#09152a",
        showrivers=False,
        showcoastlines=True,  coastlinecolor="#6090b0", coastlinewidth=0.9,
        showcountries=True,   countrycolor="#3a4a5c",   countrywidth=0.4,
        showframe=True,  framecolor="rgba(90,127,160,0.35)",
        bgcolor="#07101e",
    )

    # ── Reference scale bar inside map — South Atlantic, always ocean ────────
    # Place at lon=-40, lat=-60 (South Atlantic — never land)
    ref_lon_s, ref_lat_r = -40.0, -62.0
    ref_lon_e = ref_lon_s + wind_arrow_deg
    fig.add_trace(go.Scattergeo(
        lon=[ref_lon_s, ref_lon_e, None], lat=[ref_lat_r, ref_lat_r, None],
        mode="lines", line=dict(color="rgba(255,255,255,0.7)", width=2.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergeo(
        lon=[ref_lon_e], lat=[ref_lat_r],
        mode="markers",
        marker=dict(symbol="arrow", size=9, color="white",
                    angleref="up", angle=90, line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    # Tick marks at start and end of scale bar
    fig.add_trace(go.Scattergeo(
        lon=[ref_lon_s, ref_lon_s, None, ref_lon_e, ref_lon_e, None],
        lat=[ref_lat_r - 1.5, ref_lat_r + 1.5, None,
             ref_lat_r - 1.5, ref_lat_r + 1.5, None],
        mode="lines", line=dict(color="rgba(255,255,255,0.7)", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    # Label above scale bar
    fig.add_trace(go.Scattergeo(
        lon=[(ref_lon_s + ref_lon_e) / 2], lat=[ref_lat_r + 4.5],
        mode="text",
        text=[f"= {wind_ref:.0f} m/s"],
        textfont=dict(size=10, color="white"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=(
                "<b>Dinámica de Vientos y Corrientes</b><br>"
                "<sup>Vector Field: Vientos GEOS (U,V) + Corrientes LLC2160 (u,v)</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=14, color="#dce8f0"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.03,
            xanchor="center", x=0.5,
            font=dict(size=11, color="#dce8f0"),
            bgcolor="rgba(7,16,30,0.88)",
            bordercolor="rgba(90,127,160,0.3)", borderwidth=1,
        ),
        annotations=[
            dict(
                text=(
                    f"Escala log(P95) · 🔵 viento P95={wind_ref:.1f} m/s · "
                    f"🟠 corriente P95={current_ref:.2f} m/s"
                ),
                x=0.5, y=-0.04, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="#8a9fb0"), align="center",
            ),
        ],
        paper_bgcolor="#07101e",
        plot_bgcolor="#07101e",
        font=dict(color="#dce8f0"),
        height=560,
        margin=dict(l=0, r=0, t=60, b=45),
    )
    return fig


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
    "bubble_windstress": {
        "fn":          create_bubble_windstress,
        "label":       "🌬️ Wind Stress & SST Anomaly",
        "description": "Burbujas: tamaño = τ viento, color = anomalía SST",
    },

    "hovmoller": {
        "fn":          create_hovmoller,
        "label":       "🌡️ Hovmöller Diagram",
        "description": "Latitud × tiempo: media zonal para detectar propagación de anomalías",
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
        st.plotly_chart(fig, width="stretch")
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
            "Variable",
            options=list(VARIABLE_META.keys()),
            format_func=lambda k: f"{k} — {VARIABLE_META[k]['label']}",
            help=(
                "Select the field to visualise. "
                "Ocean variables are from the LLC2160 dataset, "
                "and atmospheric variables are from the GEOS dataset."
            ),
        )
        st.markdown("---")

        is_geos = variable.startswith("GEOS_")
        geos_face = 0
        if is_geos:
            st.markdown("### Proyección GEOS")
            geos_face = st.selectbox(
                "Cara (Face)",
                options=[0, 1, 2, 3, 4, 5],
                index=0,
                help="GEOS Atmospheric data is projected on a cubed sphere with 6 faces (0-5).",
            )
            st.markdown("---")

        # ── Time slider ───────────────────────────────────────────────────
        st.markdown("### Tiempo")
        max_time = 10269 if is_geos else (LLC2160_TOTAL_TIMESTEPS - 1)
        timestep = st.slider(
            "Timestep",
            min_value=0,
            max_value=max_time,
            value=0,
            step=1,
            help=(
                f"Timestep 0 → {max_time}. "
                "The heatmap updates automatically."
            ),
        )
        st.caption(f"≈ **{approx_datetime(timestep)}**")
        st.markdown("---")

        # ── Depth ─────────────────────────────────────────────────────────
        st.markdown("### Nivel de profundidad")
        max_depth = 51 if is_geos else (LLC2160_DEPTH_LEVELS - 1)
        depth = st.slider(
            "Depth Level",
            min_value=0,
            max_value=max_depth,
            value=0,
            step=1,
            help=f"0 = surface. GEOS has 52 levels (0-51), LLC2160 has 90 levels (0-89).",
        )
        depth_lbl = "Surface" if depth == 0 else f"Level {depth} (≈ {depth * 50} m)"
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
            "show_zonal_mean": False,
            "show_histogram":  False,
            "geos_face":       int(geos_face),
            "show_hovmoller":  False,
            "show_bubble":     True,
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


def _sample_timesteps(max_points: int = TEMP_SERIES_MAX_POINTS) -> np.ndarray:
    """Return evenly spaced timesteps across the full simulation timeline."""
    return np.unique(np.linspace(0, LLC2160_TOTAL_TIMESTEPS - 1, num=max_points, dtype=int))


def _disk_load_any_quality(variable: str, timestep: int, depth: int, face: int = 0) -> Optional[np.ndarray]:
    """Load the first cached slice found for a timestep, trying preferred qualities first."""
    for quality in (QUALITY_OPTIONS[DEFAULT_QUALITY_KEY], -6, -4, -3):
        arr = _disk_load(variable, timestep, depth, quality, face)
        if arr is not None:
            return arr
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_temperature_time_series(
    depth: int,
    quality: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    variable: str = "Theta",
    face: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mean value series for the selected variable through time, falling back to NSDF."""
    # Use variable-specific max timesteps
    max_t = 10270 if variable.startswith("GEOS_") else LLC2160_TOTAL_TIMESTEPS
    timesteps = np.unique(np.linspace(0, max_t - 1, num=TEMP_SERIES_MAX_POINTS, dtype=int))
    
    x_lo, _ = latlon_to_norm(lat_min, lon_min)
    x_hi, _ = latlon_to_norm(lat_max, lon_max)
    _, y_lo = latlon_to_norm(lat_min, lon_min)
    _, y_hi = latlon_to_norm(lat_max, lon_max)
    x_range: Tuple[float, float] = (min(x_lo, x_hi), max(x_lo, x_hi))
    y_range: Tuple[float, float] = (min(y_lo, y_hi), max(y_lo, y_hi))

    mean_values: List[float] = []
    time_values: List[datetime.datetime] = []

    for current_timestep in timesteps:
        time_values.append(_T0 + datetime.timedelta(hours=int(current_timestep)))

        temp_slice = _disk_load_any_quality(
            variable,
            int(current_timestep),
            depth,
            face,
        )

        if temp_slice is None and OPENVISUS_AVAILABLE:
            temp_slice = fetch_ocean_slice(
                variable=variable,
                timestep=int(current_timestep),
                depth=depth,
                quality=quality,
                x_range=x_range,
                y_range=y_range,
                face=face,
            )

        if temp_slice is None:
            mean_values.append(float("nan"))
            continue

        temp_arr = _mask_fill(temp_slice)
        valid = temp_arr[np.isfinite(temp_arr)]
        mean_values.append(float(np.nanmean(valid)) if valid.size else float("nan"))

    return np.array(time_values, dtype=object), np.array(mean_values, dtype=np.float32)


def create_temperature_time_series(
    times: np.ndarray,
    values: np.ndarray,
    title: str,
    variable: str = "Theta",
) -> go.Figure:
    """Line chart of mean variable value through time for the selected region."""
    valid = np.isfinite(values)
    times = times[valid]
    values = values[valid]

    meta = VARIABLE_META.get(variable, {})
    label = meta.get("label", variable)
    units = meta.get("units", "")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            line=dict(color="#ff8a5b", width=4, shape="linear"),
            name=f"{label} Trend",
            hoverinfo="skip",
            connectgaps=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="markers",
            marker=dict(size=6, color="#ff8a5b", line=dict(width=1, color="#1f232b")),
            name=f"{label} Mean",
            hovertemplate=f"Time: %{{x}}<br>Mean {label}: %{{y:.4f}} {units}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, xanchor="center", font=dict(size=12),
        ),
        xaxis=dict(
            title="Time",
            color="#b0b0b0",
            tickformat="%Y-%m-%d<br>%H:%M",
        ),
        yaxis=dict(
            title=f"Average {label} ({units})",
            color="#b0b0b0",
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        height=360,
        margin=dict(l=70, r=20, t=55, b=70),
    )
    return fig


# =============================================================================
# SECTION 7 — MAIN APPLICATION
# =============================================================================

def main() -> None:
    _configure_page()
    render_header()

    # ── 1. Sidebar controls ────────────────────────────────────────────────
    params = render_sidebar()

    # ── 2. Compute normalised coordinate ranges for OpenVisus ──────────────
    x_lo, _ = latlon_to_norm(params["lat_min"], params["lon_min"])
    x_hi, _ = latlon_to_norm(params["lat_max"], params["lon_max"])
    _, y_lo = latlon_to_norm(params["lat_min"], params["lon_min"])
    _, y_hi = latlon_to_norm(params["lat_max"], params["lon_max"])
    x_range: Tuple[float, float] = (min(x_lo, x_hi), max(x_lo, x_hi))
    y_range: Tuple[float, float] = (min(y_lo, y_hi), max(y_lo, y_hi))

    # ── 3. Fetch data ──────────────────────────────────────────────────────
    # Verificar si el dato esta en disco antes de mostrar spinner
    _key_fname = _slice_filename(
        params["variable"], params["timestep"],
        params["depth"], params["quality"],
        face=params.get("geos_face", 0)
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
            _db = _connect_dataset(params["variable"], face=params.get("geos_face", 0))
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
                    face      = params.get("geos_face", 0),
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

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,212,255,0.2);margin:10px 0 6px;'/>",
        unsafe_allow_html=True,
    )

    temp_times, temp_means = fetch_temperature_time_series(
        depth=params["depth"],
        quality=params["quality"],
        lat_min=params["lat_min"],
        lat_max=params["lat_max"],
        lon_min=params["lon_min"],
        lon_max=params["lon_max"],
        variable="Theta",
        face=0,
    )
    
    temp_title = (
        "<b>Promedio de temperatura en el tiempo</b><br>"
        f"<sup>Región actual · Profundidad {params['depth']} · Theta</sup>"
    )
    st.plotly_chart(
        create_temperature_time_series(temp_times, temp_means, temp_title, variable="Theta"),
        width="stretch",
    )

    # ── 8. Optional secondary panels (user-toggled in sidebar) ────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(0,212,255,0.2);margin:10px 0 6px;'/>",
        unsafe_allow_html=True,
    )
    with st.spinner("Cargando vectores de viento y corrientes..."):
        current_timestep = min(params["timestep"], LLC2160_TOTAL_TIMESTEPS - 1)
        wind_depth = min(params["depth"], 51)

        wind_u = fetch_ocean_slice(
            variable="GEOS_U",
            timestep=params["timestep"],
            depth=wind_depth,
            quality=params["quality"],
            x_range=x_range,
            y_range=y_range,
            face=params.get("geos_face", 0),
        )
        wind_v = fetch_ocean_slice(
            variable="GEOS_V",
            timestep=params["timestep"],
            depth=wind_depth,
            quality=params["quality"],
            x_range=x_range,
            y_range=y_range,
            face=params.get("geos_face", 0),
        )
        current_u = fetch_ocean_slice(
            variable="u",
            timestep=current_timestep,
            depth=params["depth"],
            quality=params["quality"],
            x_range=x_range,
            y_range=y_range,
            face=0,
        )
        current_v = fetch_ocean_slice(
            variable="v",
            timestep=current_timestep,
            depth=params["depth"],
            quality=params["quality"],
            x_range=x_range,
            y_range=y_range,
            face=0,
        )
        if params["variable"] in ("salt", "Theta"):
            ocean_mask_source = data
        else:
            ocean_mask_source = fetch_ocean_slice(
                variable="salt",
                timestep=current_timestep,
                depth=params["depth"],
                quality=params["quality"],
                x_range=x_range,
                y_range=y_range,
                face=0,
            )

        if wind_u is None:
            wind_u = _generate_demo_slice("GEOS_U", params["timestep"], (params["lat_min"], params["lat_max"]), (params["lon_min"], params["lon_max"]))
        if wind_v is None:
            wind_v = _generate_demo_slice("GEOS_V", params["timestep"], (params["lat_min"], params["lat_max"]), (params["lon_min"], params["lon_max"]))
        if current_u is None:
            current_u = _generate_demo_slice("u", current_timestep, (params["lat_min"], params["lat_max"]), (params["lon_min"], params["lon_max"]))
        if current_v is None:
            current_v = _generate_demo_slice("v", current_timestep, (params["lat_min"], params["lat_max"]), (params["lon_min"], params["lon_max"]))
        if ocean_mask_source is None:
            ocean_mask_source = _generate_demo_slice("salt", current_timestep, (params["lat_min"], params["lat_max"]), (params["lon_min"], params["lon_max"]))

    st.plotly_chart(
        create_wind_current_quiver(
            wind_u=wind_u,
            wind_v=wind_v,
            current_u=current_u,
            current_v=current_v,
            ocean_mask_source=ocean_mask_source,
            lats=lats,
            lons=lons,
            params=params,
        ),
        width="stretch",
    )

    secondary_panels: List[Tuple[bool, str]] = [
        (params["show_zonal_mean"], "zonal_mean"),
        (params["show_histogram"],  "histogram"),
        (params["show_hovmoller"],  "hovmoller"),
        (params["show_bubble"],     "bubble_windstress"),
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