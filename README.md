# FomOff — Data Scientist

## Deskripsi
Repository ini berisi seluruh pipeline data untuk proyek FomOff
sebagai bagian dari Coding Camp 2026 powered by DBS Foundation.
Tim ID: CC26-PSU256

## Struktur Repository
**data/processed/** — Dataset hasil olahan

**notebooks/** — Notebook analisis dan script A/B Testing

**app/** — Streamlit dashboard

**README.md** — Dokumentasi proyek

**requirements.txt** — Daftar library yang digunakan

## Dataset
Semua dataset tersedia di Google Drive berikut:
🔗 [Akses Dataset](LINK_GOOGLE_DRIVE_DI_SINI)

**Raw Dataset**
- IDX Stock Summary 2020-2024.csv — Data historis saham IDX
- Dataset-CNBCI-Sentimented.csv — Dataset berita saham berlabel sentimen

**Processed Dataset**
- dataset_berita.csv — Dataset berita hasil label encode (tersedia di repo)
- cleaned_stock_data.csv — Data saham setelah cleaning
- hasil_rsi.csv — Hasil perhitungan RSI dan Moving Average
- ab_testing_grup_a.csv — Hasil A/B Testing Grup A (RSI + MA)
- ab_testing_grup_b.csv — Hasil A/B Testing Grup B (RSI + MA + Sentimen)

## Deployed Dashboard
🔗 [Streamlit Dashboard](LINK_STREAMLIT_DI_SINI)
