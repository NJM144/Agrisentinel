# main.py
import os
import glob
import json
from typing import Optional, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform
from rasterio.transform import xy as rio_xy

import geopandas as gpd
from shapely.geometry import Polygon, Point, mapping
from shapely.validation import make_valid

import cv2
from sklearn.cluster import KMeans

# =========================================================
# Config
# =========================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")   # dossier des .tif
EPSG_WGS84 = 4326

app = FastAPI(title="AgroSentinel Soil Segmentation API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en prod: restreindre à l'origine de ton front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Modèles Pydantic
# =========================================================
class StartReq(BaseModel):
    lat: float
    lng: float
    radius_km: float = 1.0

class StartResp(BaseModel):
    jobId: str

class StatusResp(BaseModel):
    status: str = "done"
    progress: int = 100


# =========================================================
# Utilitaires
# =========================================================
def list_tifs() -> List[str]:
    """
    Liste les GeoTIFF présents dans DATA_DIR (non récursif).
    -> Décommente la version récursive si tes fichiers sont dans des sous-dossiers.
    """
    # --- NON récursif (par défaut) ---
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tif"))) + \
            sorted(glob.glob(os.path.join(DATA_DIR, "*.tiff")))
    return files

    # --- Récursif (optionnel) ---
    # patterns = ["**/*.tif", "**/*.tiff"]
    # files = []
    # for p in patterns:
    #     files += glob.glob(os.path.join(DATA_DIR, p), recursive=True)
    # return sorted(files)

def bounds_wgs84(src) -> Tuple[float, float, float, float]:
    """Retourne (min_lon, min_lat, max_lon, max_lat) des bounds du raster en WGS84."""
    b = src.bounds
    xs = [b.left, b.right]
    ys = [b.bottom, b.top]
    lon, lat = rio_transform(src.crs, f"EPSG:{EPSG_WGS84}", xs, ys)
    return min(lon), min(lat), max(lon), max(lat)

def point_in_tif(tif_path: str, lat: float, lng: float) -> bool:
    with rasterio.open(tif_path) as src:
        min_lon, min_lat, max_lon, max_lat = bounds_wgs84(src)
        return (min_lon <= lng <= max_lon) and (min_lat <= lat <= max_lat)

def find_tif_for_point(lat: float, lng: float) -> Optional[str]:
    for tif in list_tifs():
        try:
            if point_in_tif(tif, lat, lng):
                return tif
        except Exception:
            continue
    return None

def read_band(src, band_index: int, out_shape: Tuple[int, int]):
    return src.read(band_index, out_shape=out_shape, resampling=Resampling.bilinear).astype("float64")

def read_band_safe(src, band_index: int, out_shape: Tuple[int, int]):
    try:
        return read_band(src, band_index, out_shape)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Impossible de lire la bande index={band_index}. "
                f"Vérifie que le GeoTIFF contient les bandes attendues (B2=2, B3=3, B4=4, B8=8, B11=11). "
                f"Erreur interne: {e}"
            )
        )

def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float64")
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def safe_unary_union(gdf: gpd.GeoDataFrame):
    """Nettoie et fusionne des géométries potentiellement invalides."""
    if gdf is None or gdf.empty:
        return None
    cleaned = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = make_valid(geom)
            geom = geom.buffer(0)
            if geom.is_valid and not geom.is_empty:
                cleaned.append(geom)
        except Exception:
            continue
    if not cleaned:
        return None
    return gpd.GeoSeries(cleaned, crs=gdf.crs).unary_union


# =========================================================
# Coeur de l'algo (3 masques)
# =========================================================
def run_segmentation_on_tif(tif_path: str, lat: float, lng: float, radius_km: float) -> dict:
    epsilon = 1e-6

    # 1) Lecture des bandes
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        out_shape = (H, W)

        # Indices Sentinel-2 classiques
        # B2 (blue)=2, B3 (green)=3, B4 (red)=4, B8 (NIR)=8, B11 (SWIR)=11
        blue  = read_band_safe(src, 2, out_shape)
        green = read_band_safe(src, 3, out_shape)
        red   = read_band_safe(src, 4, out_shape)
        nir   = read_band_safe(src, 8, out_shape)
        swir  = read_band_safe(src, 11, out_shape)

        transform_affine = src.transform
        crs = src.crs

    # 2) Buffer autour du point (dans le CRS du raster)
    point_wgs = gpd.GeoDataFrame(geometry=[Point(lng, lat)], crs=f"EPSG:{EPSG_WGS84}")
    point_img = point_wgs.to_crs(crs)  # reprojeter dans le CRS du raster
    buffer_img = point_img.buffer(radius_km * 1000).to_crs(crs)  # rayon en mètres

    # 3) Indices de végétation
    NDVI = (nir - red) / (nir + red + epsilon)
    NDWI = (nir - swir) / (nir + swir + epsilon)
    BSI  = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + epsilon)
    SAVI = (1.5 * (nir - red)) / (nir + red + 0.5 + epsilon)

    NDVI = normalize01(NDVI)
    NDWI = normalize01(NDWI)
    BSI  = normalize01(BSI)
    SAVI = normalize01(SAVI)

    # 4) Clustering non supervisé (3 classes)
    features = np.stack([NDVI, NDWI, BSI, SAVI], axis=-1).reshape(-1, 4)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features).reshape(H, W)

    # Ordonner par intensité NDVI croissante :
    # 0 -> sol nu (NDVI faible)
    # 1 -> culture (NDVI moyen)
    # 2 -> forêt (NDVI fort)
    cluster_means = [NDVI[labels == i].mean() if np.any(labels == i) else -1 for i in range(3)]
    order = np.argsort(cluster_means)
    final_map = np.zeros_like(labels)
    for new_label, old_label in enumerate(order):
        final_map[labels == old_label] = new_label

    # 5) Masque spatial (en dehors du buffer = -1)
    rows, cols = np.indices(final_map.shape)
    xs, ys = rio_xy(transform_affine, rows, cols)
    xs = np.array(xs); ys = np.array(ys)
    points_grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()),
        crs=crs
    )
    mask_spatial = points_grid.within(buffer_img.unary_union).values.reshape(final_map.shape)
    final_map[~mask_spatial] = -1

    # 6) Extraction des polygones par classe via contours OpenCV
    label_to_name = {0: "Zone nue", 1: "Zone cultivée", 2: "Zone forestière"}
    api_class_map = {"Zone nue": "bare_soil", "Zone cultivée": "crop", "Zone forestière": "forest"}

    gdf_classes = {}
    for class_value, class_name in label_to_name.items():
        mask = (final_map == class_value).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for c in contours:
            if cv2.contourArea(c) > 500:  # filtre petites taches
                coords = []
                for pt in c[:, 0]:
                    row, col = int(pt[1]), int(pt[0])
                    x, y = rio_xy(transform_affine, row, col)
                    coords.append((x, y))
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))

        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs).to_crs(epsg=EPSG_WGS84)
            gdf_m = gdf.to_crs(epsg=3857)
            gdf["surface_ha"] = gdf_m.area / 10_000
            gdf_classes[class_name] = gdf
        else:
            gdf_classes[class_name] = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{EPSG_WGS84}")

    # 7) Construire la FeatureCollection
    features_out = []
    for cname, gdf in gdf_classes.items():
        if gdf.empty:
            continue
        merged = safe_unary_union(gdf)
        if merged is None or merged.is_empty:
            continue
        gj = mapping(merged)  # (Multi)Polygon
        features_out.append({
            "type": "Feature",
            "properties": {"class": api_class_map[cname]},
            "geometry": gj
        })

    return {"type": "FeatureCollection", "features": features_out}


# =========================================================
# Endpoints
# =========================================================
@app.post("/api/soil-segmentation", response_model=StartResp)
def start(req: StartReq):
    # 0) Vérifier la présence de fichiers
    files = list_tifs()
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun fichier GeoTIFF trouvé dans DATA_DIR='{os.environ.get('DATA_DIR','data')}'. "
                   "Placez des .tif/.tiff dans ce dossier ou corrigez DATA_DIR."
        )

    # 1) Trouver un TIF qui couvre le point
    tif_path = find_tif_for_point(req.lat, req.lng)
    if not tif_path:
        # Construire un message explicite listant les emprises disponibles
        examples = []
        for p in files[:5]:
            try:
                with rasterio.open(p) as src:
                    minlon, minlat, maxlon, maxlat = bounds_wgs84(src)
                    examples.append({
                        "path": p,
                        "bounds_wgs84": [round(minlon,5), round(minlat,5), round(maxlon,5), round(maxlat,5)]
                    })
            except Exception:
                continue

        raise HTTPException(
            status_code=404,
            detail={
                "message": "Aucune image TIFF ne couvre le point demandé.",
                "point_wgs84": {"lat": req.lat, "lng": req.lng},
                "DATA_DIR": os.environ.get("DATA_DIR", "data"),
                "fichiers_trouves": len(files),
                "exemples_emprises": examples
            }
        )
    # 2) Encodage synchrone en jobId
    job_id = json.dumps({"tif": tif_path, "lat": req.lat, "lng": req.lng, "r": req.radius_km})
    return StartResp(jobId=job_id)

@app.get("/api/soil-segmentation/{job_id}/status", response_model=StatusResp)
def status(job_id: str):
    # Pas de vraie queue dans ce POC : job terminé immédiatement
    return StatusResp(status="done", progress=100)

@app.get("/api/soil-segmentation/{job_id}/result")
def result(job_id: str):
    try:
        payload = json.loads(job_id)
        tif = payload["tif"]
        lat = float(payload["lat"])
        lng = float(payload["lng"])
        rkm = float(payload["r"])
    except Exception:
        raise HTTPException(status_code=400, detail="job_id invalide")

    if not os.path.exists(tif):
        raise HTTPException(
            status_code=404,
            detail=f"Fichier TIFF introuvable côté serveur: '{tif}'."
        )

    fc = run_segmentation_on_tif(tif, lat, lng, rkm)
    # S'assurer qu'on renvoie bien 3 classes max
    fc["features"] = [f for f in fc.get("features", []) if f.get("properties", {}).get("class") in {"bare_soil", "crop", "forest"}]
    return fc


# =========================================================
# Endpoint de debug (diagnostic rapide)
# =========================================================
@app.get("/api/debug/files")
def debug_files():
    paths = list_tifs()
    out = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                minlon, minlat, maxlon, maxlat = bounds_wgs84(src)
                out.append({
                    "path": p,
                    "crs": str(src.crs),
                    "bounds_wgs84": [minlon, minlat, maxlon, maxlat],
                    "bands": src.count
                })
        except Exception as e:
            out.append({"path": p, "error": str(e)})
    return {
        "DATA_DIR": os.environ.get("DATA_DIR", "data"),
        "count": len(paths),
        "files": out
    }

# =========================================================
# Lancement (dev):
# uvicorn main:app --reload --port 8000
# Avec data ailleurs:
# DATA_DIR=/chemin/vers/data uvicorn main:app --reload --port 8000
# =========================================================
