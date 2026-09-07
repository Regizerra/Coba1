"""
PETA POTENSI TIMBULAN AKTIVITAS MASYARAKAT — GRID HEKSAGON
============================================================
Mengagregasi data POI (titik, dengan kategori) dan populasi (poligon
per kecamatan) ke dalam grid heksagon, lalu menghitung skor komposit
"potensi aktivitas" per sel.

Input:
  - POI_Aktivitas_Masyarakat.geojson       (Point, punya field KATEGORI)
  - Jumlah_Penduduk_Kota_Yogyakarta.geojson (MultiPolygon per kecamatan, field Penduduk)
Output:
  - Heksagon_Potensi_Aktivitas_Yogyakarta.geojson

Dependensi (install dulu kalau belum ada):
    pip install geopandas shapely pyproj fiona pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# ==================================================================
# 0. KONFIGURASI — bagian ini yang paling sering perlu diubah
# ==================================================================
POI_PATH =r"C:\Users\REGINALD\Downloads\POI Aktivitas Masyarakat.geojson"
POP_PATH =r"C:\Users\REGINALD\Downloads\Jumlah Penduduk Kota Yogyakarta.geojson"
OUT_PATH ="Heksagon_Potensi_Aktivitas_Yogyakarta.geojson"

# UTM zone 49S -> cocok untuk wilayah ±108-114 BT belahan bumi selatan
# (termasuk Yogyakarta, ~110.3 BT). Dipakai supaya luas & jarak bisa
# dihitung dalam meter, bukan derajat (geojson aslinya WGS84/EPSG:4326,
# dan derajat itu bukan satuan jarak yang seragam).
UTM_CRS = 32749

# "Circumradius" heksagon = jarak dari pusat ke tiap titik sudutnya, dalam meter.
# Semakin kecil R -> semakin banyak & semakin kecil sel heksagonnya
# -> gradasi visual makin halus, tapi POI per sel makin sedikit (makin "noisy").
# R=510 -> ~70 sel (skala grid lama tim). R=200 -> ~378 sel (versi lebih halus).
HEX_R = 200.0

# Mapping nilai kolom KATEGORI di data POI -> nama kolom breakdown di output.
# Sesuaikan kalau nama/isi kategori di data kalian beda.
KATEGORI_MAP = {
    "Kuliner": "poi_kuliner",
    "Peribadatan": "poi_peribadatan",
    "Kesehatan": "poi_kesehatan",
    "Pendidikan": "poi_pendidikan",
    "Perkantoran": "poi_perkantoran",
    "Pariwisata": "poi_pariwisata",
    "Perdagangan": "poi_perdagangan",
    "Transportasi": "poi_transportasi",
}


# ==================================================================
# 1. FUNGSI PEMBUAT HEKSAGON
# ==================================================================
def buat_heksagon(cx, cy, R):
    """Satu heksagon reguler orientasi 'pointy-top' (sudut runcing di
    atas & bawah), berpusat di (cx, cy), dengan circumradius R."""
    sudut = np.radians(30 + 60 * np.arange(6))   # 30,90,150,210,270,330 derajat
    xs = cx + R * np.cos(sudut)
    ys = cy + R * np.sin(sudut)
    return Polygon(zip(xs, ys))


def buat_grid_heksagon(bounds, R, pad_rings=1):
    """Tessellation penuh heksagon (saling mengunci tanpa celah/tumpang
    tindih) yang menutupi kotak `bounds` = (minx, miny, maxx, maxy).
    `pad_rings` menambah sedikit ekstra di tepi supaya area tepi ikut
    tertutup penuh sebelum nanti difilter ke batas wilayah."""
    minx, miny, maxx, maxy = bounds
    jarak_h = np.sqrt(3) * R      # jarak antar-pusat heksagon, dalam 1 baris
    jarak_v = 1.5 * R             # jarak antar-baris
    pad = pad_rings * max(jarak_h, jarak_v)
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    daftar_poligon = []
    baris = 0
    y = miny
    while y - R <= maxy:
        # baris ganjil digeser setengah `jarak_h` biar antar-baris saling mengunci
        offset_x = (jarak_h / 2) if (baris % 2 == 1) else 0.0
        x = minx + offset_x
        while x - R <= maxx:
            daftar_poligon.append(buat_heksagon(x, y, R))
            x += jarak_h
        y += jarak_v
        baris += 1
    return daftar_poligon


# ==================================================================
# 2. BACA DATA & REPROJECT KE UTM (satuan meter)
# ==================================================================
poi = gpd.read_file(POI_PATH).to_crs(epsg=UTM_CRS)
pop = gpd.read_file(POP_PATH).to_crs(epsg=UTM_CRS)

batas_kota = pop.union_all()      # dissolve semua poligon kecamatan -> 1 batas kota
bounds = pop.total_bounds         # (minx, miny, maxx, maxy)


# ==================================================================
# 3. GENERATE GRID, SIMPAN SEL YANG BERSINGGUNGAN DENGAN BATAS KOTA
# ==================================================================
# Catatan: heksagon TIDAK dipotong mengikuti batas kota (bentuknya
# tetap heksagon utuh) — ini praktik standar untuk hexbin map, supaya
# semua sel tetap punya luas yang sama & bisa dibandingkan adil.
# Konsekuensinya, sel di pinggir kota nilainya bisa sedikit "diredam"
# karena sebagian areanya jatuh di luar kota (memang tidak ada datanya).
poligon_hex = buat_grid_heksagon(bounds, HEX_R)
hexes = gpd.GeoDataFrame(geometry=poligon_hex, crs=poi.crs)
hexes = hexes[hexes.intersects(batas_kota)].reset_index(drop=True)
hexes["hex_id"] = hexes.index
hexes["luas_km2"] = hexes.geometry.area / 1e6
print(f"Jumlah sel heksagon: {len(hexes)}")


# ==================================================================
# 4. JUMLAH POI PER SEL (spatial join titik-dalam-poligon)
# ==================================================================
join = gpd.sjoin(poi, hexes[["hex_id", "geometry"]], predicate="within", how="inner")

total = join.groupby("hex_id").size().rename("poi_total")
hexes = hexes.merge(total, on="hex_id", how="left")
hexes["poi_total"] = hexes["poi_total"].fillna(0).astype(int)

# breakdown per kategori: pivot baris=hex_id, kolom=KATEGORI
per_kategori = join.groupby(["hex_id", "KATEGORI"]).size().unstack(fill_value=0)
per_kategori = per_kategori.rename(columns=KATEGORI_MAP)
for kolom in KATEGORI_MAP.values():        # jaga-jaga kalau ada kategori yg 0 di semua sel
    if kolom not in per_kategori.columns:
        per_kategori[kolom] = 0
hexes = hexes.merge(per_kategori[list(KATEGORI_MAP.values())], on="hex_id", how="left")
for kolom in KATEGORI_MAP.values():
    hexes[kolom] = hexes[kolom].fillna(0).astype(int)

# sanity check: jumlah semua kategori harus sama dengan poi_total
asumsi_benar = (hexes[list(KATEGORI_MAP.values())].sum(axis=1) == hexes["poi_total"]).all()
assert asumsi_benar, "breakdown kategori tidak sinkron dengan poi_total!"


# ==================================================================
# 5. INTERPOLASI POPULASI KE TIAP SEL (areal weighting)
# ==================================================================
# Data populasi cuma tersedia per kecamatan (poligon besar), bukan per
# titik. Supaya bisa dipecah ke tiap sel heksagon (poligon kecil),
# dipakai asumsi: kepadatan penduduk MERATA di dalam satu kecamatan.
# Untuk tiap sel heksagon:
#   penduduk_sel = SUM( kepadatan_kecamatan_i x luas_irisan(sel, kecamatan_i) )
#                  untuk semua kecamatan i yang beririsan dgn sel itu
# Ini metode standar disebut "areal interpolation" / areal weighting.
pop["luas_kec_m2"] = pop.geometry.area
pop["kepadatan"] = pop["Penduduk"] / pop["luas_kec_m2"]     # jiwa per m2

irisan = gpd.overlay(hexes[["hex_id", "geometry"]],
                      pop[["kepadatan", "geometry"]], how="intersection")
irisan["luas_irisan_m2"] = irisan.geometry.area
irisan["kontribusi"] = irisan["kepadatan"] * irisan["luas_irisan_m2"]

penduduk_per_sel = irisan.groupby("hex_id")["kontribusi"].sum().rename("penduduk")
hexes = hexes.merge(penduduk_per_sel, on="hex_id", how="left")
hexes["penduduk"] = hexes["penduduk"].fillna(0).round(0).astype(int)

# sanity check: total penduduk hasil interpolasi harus ~sama dgn total sumber
print(f"Cek konservasi populasi -> sumber: {pop['Penduduk'].sum()}, "
      f"hasil interpolasi: {hexes['penduduk'].sum()}")


# ==================================================================
# 6. DENSITAS -> NORMALISASI (0-1) -> SKOR KOMPOSIT
# ==================================================================
hexes["densitas_poi"] = hexes["poi_total"] / hexes["luas_km2"]
hexes["densitas_penduduk"] = hexes["penduduk"] / hexes["luas_km2"]


def normalisasi_minmax(s):
    """Skala nilai ke rentang 0-1: (nilai - min) / (max - min)."""
    rentang = s.max() - s.min()
    return (s - s.min()) / rentang if rentang > 0 else s * 0


hexes["skor_poi"] = normalisasi_minmax(hexes["densitas_poi"]).round(3)
hexes["skor_penduduk"] = normalisasi_minmax(hexes["densitas_penduduk"]).round(3)

# Skor komposit = rata-rata sederhana (bobot POI:populasi = 50:50).
# Mau bobot beda? tinggal ganti baris ini, misal POI dianggap lebih
# penting (60:40):
#   hexes["skor_aktivitas"] = 0.6*hexes["skor_poi"] + 0.4*hexes["skor_penduduk"]
hexes["skor_aktivitas"] = ((hexes["skor_poi"] + hexes["skor_penduduk"]) / 2).round(3)


# ==================================================================
# 7. KLASIFIKASI 5 KELAS BERDASARKAN KUANTIL
# ==================================================================
# qcut membagi data jadi 5 kelompok dengan JUMLAH SEL yang kurang lebih
# sama rata (bukan rentang skor yang sama rata seperti equal-interval),
# jadi lebih tahan terhadap data yang miring/skewed.
label_kelas = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
hexes["kelas_aktivitas"] = pd.qcut(
    hexes["skor_aktivitas"], q=5, labels=label_kelas, duplicates="drop"
).astype(str)


# ==================================================================
# 8. RAPIKAN ANGKA, REPROJECT BALIK KE WGS84, EXPORT GEOJSON
# ==================================================================
hexes["luas_km2"] = hexes["luas_km2"].round(4)
hexes["densitas_poi"] = hexes["densitas_poi"].round(2)
hexes["densitas_penduduk"] = hexes["densitas_penduduk"].round(1)

urutan_kolom = (
    ["hex_id", "luas_km2", "poi_total"] + list(KATEGORI_MAP.values()) +
    ["penduduk", "densitas_poi", "densitas_penduduk",
     "skor_poi", "skor_penduduk", "skor_aktivitas", "kelas_aktivitas", "geometry"]
)

# GeoJSON RFC 7946 mensyaratkan WGS84 (EPSG:4326) -> reproject balik dari UTM
hasil = hexes.to_crs(epsg=4326)[urutan_kolom]
hasil.to_file(OUT_PATH, driver="GeoJSON")
print(f"\nSelesai -> {OUT_PATH} ({len(hasil)} sel)")
print(hasil.drop(columns="geometry").describe())


# ==================================================================
# 9. BONUS: PREVIEW CEPAT (opsional, buat cek visual sebelum ke MAPID)
# ==================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 9))
hasil.plot(column="skor_aktivitas", ax=ax, cmap="BuGn", edgecolor="none", legend=True)
pop.to_crs(epsg=4326).boundary.plot(ax=ax, color="#6a6a6a", linewidth=0.8, linestyle="--")
ax.set_title(f"Potensi Aktivitas Masyarakat — {len(hasil)} sel heksagon")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("preview_aktivitas.png", dpi=180)
print("Preview disimpan -> preview_aktivitas.png")
