"""
download_data.py — Pre-descarga local del dataset LLC2160 (NASA DYAMOND)
========================================================================

Ejecutar ANTES de abrir la app:
    python download_data.py

La app leerá los datos del disco (<0.1 s) en vez de la red (5-15 s).

Clave de archivo: <variable>_t<TTTTT>_d<D>_q<Q>_global.npz
  Siempre guarda el slice GLOBAL. La app recorta la region en memoria,
  lo que elimina el problema de claves no coincidentes con hashes.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

try:
    import OpenVisus as ov
except ImportError:
    print("ERROR: instala OpenVisus:  pip install OpenVisus")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download")

# =============================================================================
# CONFIGURA AQUI que descargar
# =============================================================================

VARIABLES   = ["salt", "Theta", "u", "v", "w", "GEOS_U", "GEOS_V", "GEOS_P", "GEOS_T"]
TIMESTEPS   = list(range(0, 10))   # ampliar segun necesidad
DEPTH       = 0
QUALITY     = -5    # -6 muy rapido | -5 rapido (~256px) | -4 medio (~512px)
MAX_WORKERS = 4

# =============================================================================
# URLs (NO tocar - deben coincidir con app.py)
# =============================================================================

_NSDF_BASE = (
    "https://nsdf-climate3-origin.nationalresearchplatform.org:50098"
    "/nasa/nsdf/climate3/dyamond/"
)
DATASET_URLS = {
    "salt":  _NSDF_BASE + "mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
    "Theta": _NSDF_BASE + "mit_output/llc2160_theta/llc2160_theta.idx",
    "u":     _NSDF_BASE + "mit_output/llc2160_arco/visus.idx",
    "v":     _NSDF_BASE + "mit_output/llc2160_v/v_llc2160_x_y_depth.idx",
    "w":     _NSDF_BASE + "mit_output/llc2160_w/llc2160_w.idx",
}

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SLICE_DIR    = os.path.join(_PROJECT_ROOT, ".visus-cache", "slices")
_VISUS_CACHE = os.path.join(_PROJECT_ROOT, ".visus-cache", "dyamond-ocean")
os.makedirs(SLICE_DIR, exist_ok=True)
os.makedirs(_VISUS_CACHE, exist_ok=True)
os.environ.setdefault("VISUS_CACHE", _VISUS_CACHE)

# =============================================================================
# Clave de archivo (IDENTICA a app.py — no cambiar)
# =============================================================================

def slice_filename(variable: str, timestep: int, depth: int, quality: int, face: int = 0) -> str:
    if variable.startswith("GEOS_"):
        return f"{variable}_face{face}_t{timestep:05d}_d{depth}_q{quality}_global.npz"
    return f"{variable}_t{timestep:05d}_d{depth}_q{quality}_global.npz"

# =============================================================================
# Descarga
# =============================================================================

_db_cache: Dict[str, object] = {}

def _get_db(variable: str, face: int = 0):
    key = (variable, face) if variable.startswith("GEOS_") else variable
    if key not in _db_cache:
        if variable.startswith("GEOS_"):
            var_char = variable.split("_")[1].lower()
            url = f"https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/GEOS/GEOS_{var_char.upper()}/{var_char}_face_{face}_depth_52_time_0_10269.idx"
        else:
            url = DATASET_URLS[variable]
        _db_cache[key] = ov.LoadDataset(url)
    return _db_cache[key]


def download_one(variable: str, timestep: int, depth: int, quality: int, face: int = 0) -> str:
    fname = slice_filename(variable, timestep, depth, quality, face)
    path  = os.path.join(SLICE_DIR, fname)
    if os.path.exists(path):
        return "skip"
    try:
        db  = _get_db(variable, face)
        raw = db.read(time=timestep, quality=quality, z=[depth, depth + 1])
        if raw is None:
            return "error"
        arr = np.array(raw, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr[0]
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, arr=arr)
        os.replace(tmp, path)
        return "ok"
    except Exception as exc:
        log.error("ERROR var=%s t=%d: %s", variable, timestep, exc)
        try:
            os.remove(path + ".tmp")
        except OSError:
            pass
        return "error"

# =============================================================================
# Main
# =============================================================================

def main():
    # jobs are tuples of (variable, timestep, face)
    jobs    = [(v, t, 0) for v in VARIABLES for t in TIMESTEPS]
    pending = [(v, t, f) for v, t, f in jobs
               if not os.path.exists(
                   os.path.join(SLICE_DIR, slice_filename(v, t, DEPTH, QUALITY, f)))]
    already = len(jobs) - len(pending)

    qname = {-6: "~128px", -5: "~256px", -4: "~512px"}.get(QUALITY, str(QUALITY))
    print()
    print("=" * 60)
    print("  DYAMOND dataset (Ocean LLC2160 & GEOS Atmos) - descarga local")
    print("=" * 60)
    print(f"  Variables : {', '.join(VARIABLES)}")
    print(f"  Timesteps : {TIMESTEPS[0]}-{TIMESTEPS[-1]}  ({len(TIMESTEPS)} pasos)")
    print(f"  Calidad   : {QUALITY} ({qname})")
    print(f"  Directorio: {SLICE_DIR}")
    print(f"  Total: {len(jobs)}  |  en disco: {already}  |  a descargar: {len(pending)}")
    print("=" * 60)

    if not pending:
        print("\nTodo listo. Abre la app:")
        print("   streamlit run app.py\n")
        return

    t0 = time.time()
    ok = fail = done = 0

    def _job(args):
        v, t, f = args
        tj = time.time()
        r  = download_one(v, t, DEPTH, QUALITY, f)
        return v, t, f, r, time.time() - tj

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_job, j) for j in pending]
        for fut in concurrent.futures.as_completed(futures):
            v, t, f, r, elapsed = fut.result()
            done += 1
            if r == "ok":    ok   += 1
            if r == "error": fail += 1
            icon = "OK  " if r == "ok" else ("SKIP" if r == "skip" else "FAIL")
            pct  = done / len(pending) * 100
            face_str = f" face={f}" if v.startswith("GEOS_") else ""
            print(f"  {icon}  [{done:3d}/{len(pending)}  {pct:5.1f}%]  "
                  f"{v:<6}{face_str} t={t:<4d}  {elapsed:.1f}s")

    print()
    print("=" * 60)
    print(f"  Total: {time.time()-t0:.0f}s  OK: {ok}  Fallidos: {fail}")
    print("=" * 60)
    if fail == 0:
        print("\nDescarga completa. Abre la app:")
        print("   streamlit run app.py\n")
    else:
        print(f"\n{fail} fallo(s). Vuelve a ejecutar para reintentar.\n")


if __name__ == "__main__":
    main()
