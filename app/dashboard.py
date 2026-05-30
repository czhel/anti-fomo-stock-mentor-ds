import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="INVESTSENSE AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"   # tetap expanded awal
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sembunyikan menu utama dan footer */
#MainMenu, footer { visibility: hidden; }

/* MENYEMBUNYIKAN IKON ANCHOR LINK (RANTAI) AGAR URL TIDAK MACET */
h1 a, h2 a, h3 a {
    display: none !important;
}

/* =========================================================
   STYLING HALAMAN UTAMA (DIUBAH JADI PUTIH)
   ========================================================= */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
}
[data-testid="stMain"] {
    background-color: #ffffff !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

h1 {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 500 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem !important;
}
h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.35rem !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* =========================================================
   STYLING SIDEBAR (DIUBAH JADI HITAM & NAVIGASI ELEGAN)
   ========================================================= */
[data-testid="stSidebar"] {
    background: #000000 !important;
    border-right: 1px solid #262626;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important; 
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #e5e5e5 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
/* Efek hover item navigasi di latar belakang hitam */
[data-testid="stSidebar"] .stRadio label:hover {
    background: #262626 !important;
    color: #ffffff !important;
}

hr {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.875rem !important;
}

.box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.box-blue   { background:#f8fafc; border-left:3px solid #3b82f6; color:#1e40af; border-top:1px solid #e2e8f0; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0; }
.box-green  { background:#f0fdf4; border-left:3px solid #22c55e; color:#166534; }
.box-yellow { background:#fefce8; border-left:3px solid #eab308; color:#854d0e; }
.box-red    { background:#fef2f2; border-left:3px solid #ef4444; color:#991b1b; }
.box-gray   { background:#f8fafc; border-left:3px solid #94a3b8; color:#475569; }

[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

.stCaption p {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {
    'modeBarButtonsToRemove': [
        'zoom2d','pan2d','select2d','lasso2d',
        'autoScale2d','resetScale2d','toggleHover'
    ],
    'displaylogo': False,
    'scrollZoom': False
}

PLOTLY_LAYOUT = dict(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(family='DM Sans', color='#374151'),
    margin=dict(t=48, b=32, l=16, r=16),
    xaxis=dict(showgrid=True, gridcolor='#f1f5f9',
               linecolor='#e2e8f0', tickfont=dict(size=11)),
    yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
               linecolor='#e2e8f0', tickfont=dict(size=11))
)

SIGNAL_ORDER = [
    'Strong Buy',
    'Buy',
    'Hold',
    'Sell',
    'Strong Sell'
]

SIGNAL_COLORS = {
    'Strong Buy': '#166534',
    'Buy': '#22c55e',
    'Hold': '#94a3b8',
    'Sell': '#ef4444',
    'Strong Sell': '#7f1d1d'
}

SENTIMENT_COLORS = {
    'positif': '#22c55e',
    'netral': '#94a3b8',
    'negatif': '#ef4444'
}

@st.cache_data
def load_saham():
    return pd.read_csv(
        '../data/processed/cleaned_stock_data.csv',
        parse_dates=['Last Trading Date']
    )

@st.cache_data
def load_grup_a():
    return pd.read_csv(
        '../data/processed/ab_testing_grup_a.csv',
        parse_dates=['Last Trading Date']
    )

@st.cache_data
def load_grup_b():
    return pd.read_csv(
        '../data/processed/ab_testing_grup_b.csv',
        parse_dates=['Last Trading Date']
    )

@st.cache_data
def load_berita():
    return pd.read_csv('../data/processed/dataset_berita.csv')

df_saham  = load_saham()
df_grup_a = load_grup_a()
df_grup_b = load_grup_b()

try:
    df_berita     = load_berita()
    berita_loaded = True
except Exception:
    berita_loaded = False

HALAMAN_EDA      = "EDA Dataset Saham"
HALAMAN_SENTIMEN = "Distribusi Sentimen Berita"
HALAMAN_AB       = "Hasil A/B Testing"

# --- LOGIK SINKRONISASI URL QUERY PARAMS ---
if "page" in st.query_params:
    halaman_saat_ini = st.query_params["page"]
    daftar_opsi = [HALAMAN_EDA, HALAMAN_SENTIMEN, HALAMAN_AB]
    idx_default = daftar_opsi.index(halaman_saat_ini) if halaman_saat_ini in daftar_opsi else 0
else:
    idx_default = 0

with st.sidebar:
    st.markdown("""
    <div style='font-family: "DM Mono", monospace; 
                font-size: 1.35rem; 
                font-weight: 500; 
                letter-spacing: 0.15em; 
                color: #ffffff; 
                text-align: center; 
                padding: 15px 0 5px 0;'>
        INVESTSENSE AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 0px 8px 20px; 
                border-bottom: 1px solid #262626; 
                margin-bottom: 8px'>
        <div style='font-size: 0.65rem; color: #a3a3a3;
                    margin-top: 6px; letter-spacing: 0.06em;
                    text-transform: uppercase; text-align: center;'>
            Data Science Dashboard
        </div>
        <div style='font-size: 0.6rem; color: #525252; 
                    margin-top: 2px; text-align: center;'>
            CC26-PSU256
        </div>
    </div>
    """, unsafe_allow_html=True)

    halaman = st.radio(
        label="Navigasi", 
        options=[HALAMAN_EDA, HALAMAN_SENTIMEN, HALAMAN_AB],
        index=idx_default, # Menggunakan index default dari URL
        key="navigasi_utama",
        label_visibility="collapsed" 
    )

    # Simpan halaman terbaru ke parameter URL browser setiap ada perubahan click
    st.query_params["page"] = halaman

    st.markdown("""
    <div style='margin-top: 32px; padding-top: 16px;
                border-top: 1px solid #262626'>
        <div style='font-size: 0.65rem; color: #a3a3a3; 
                    text-align: center; line-height: 1.6'>
            Coding Camp 2026<br><span style='color:#737373'>powered by DBS Foundation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if halaman == HALAMAN_EDA:

    st.title("EDA Dataset Saham IDX")
    st.markdown(
        "Eksplorasi data saham IDX periode **September 2022 – Oktober 2023** "
        "sebagai dasar analisis teknikal proyek INVESTSENSE AI."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data",    f"{len(df_saham):,} baris")
    c2.metric("Jumlah Saham",  f"{df_saham['Stock Code'].nunique():,} saham")
    c3.metric("Periode Awal",  str(df_saham['Last Trading Date'].min().date()))
    c4.metric("Periode Akhir", str(df_saham['Last Trading Date'].max().date()))

    st.markdown("---")

    st.subheader("Pergerakan Harga Saham")
    st.markdown(
        "Ketik atau pilih kode saham untuk melihat grafik "
        "pergerakan harga beserta sinyal teknikal."
    )

    stock_list     = sorted(df_saham['Stock Code'].unique().tolist())
    selected_stock = st.selectbox(
        "Kode Saham",
        options=stock_list,
        index=stock_list.index('GOTO') if 'GOTO' in stock_list else 0,
        placeholder="Ketik kode saham..."
    )

    df_stock = df_saham[df_saham['Stock Code'] == selected_stock].copy()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_stock['Last Trading Date'], y=df_stock['Close'],
        mode='lines', name='Close',
        line=dict(color='#3b82f6', width=2)
    ))
    if 'MA5' in df_stock.columns:
        fig1.add_trace(go.Scatter(
            x=df_stock['Last Trading Date'], y=df_stock['MA5'],
            mode='lines', name='MA5',
            line=dict(color='#f59e0b', width=1.2, dash='dot')
        ))
    if 'MA20' in df_stock.columns:
        fig1.add_trace(go.Scatter(
            x=df_stock['Last Trading Date'], y=df_stock['MA20'],
            mode='lines', name='MA20',
            line=dict(color='#ef4444', width=1.2, dash='dash')
        ))

    buy_signal  = df_stock[df_stock['MA_Crossover'] == 1]
    sell_signal = df_stock[df_stock['MA_Crossover'] == -1]
    fig1.add_trace(go.Scatter(
        x=buy_signal['Last Trading Date'], y=buy_signal['Close'],
        mode='markers', name='Sinyal Beli',
        marker=dict(symbol='triangle-up', size=10, color='#22c55e')
    ))
    fig1.add_trace(go.Scatter(
        x=sell_signal['Last Trading Date'], y=sell_signal['Close'],
        mode='markers', name='Sinyal Jual',
        marker=dict(symbol='triangle-down', size=10, color='#ef4444')
    ))

    fig1.update_layout(
        **PLOTLY_LAYOUT,
        title=f'{selected_stock} — Harga Penutupan & Sinyal MA',
        xaxis_title='Tanggal', yaxis_title='Harga (IDR)',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    <div class='box box-blue'>
    <b>Cara membaca grafik:</b> Sinyal beli ditunjukkan oleh segitiga hijau (Golden Cross), sedangkan sinyal jual oleh segitiga merah (Death Cross). Lintasan MA5 dan MA20 menggambarkan rata-rata pergerakan harga 5 dan 20 hari terakhir.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Distribusi RSI")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        df_rsi    = df_saham.dropna(subset=['RSI'])
        hist_data = [df_rsi['RSI'].values]
        fig2 = ff.create_distplot(
            hist_data, ['Distribusi RSI'],
            show_hist=False, show_rug=False,
            colors=['#6366f1']
        )
        fig2.add_vline(x=30, line_dash='dash', line_color='#22c55e',
                       line_width=1.5, annotation_text='Oversold (30)',
                       annotation_position='top left')
        fig2.add_vline(x=70, line_dash='dash', line_color='#ef4444',
                       line_width=1.5, annotation_text='Overbought (70)',
                       annotation_position='top right')
        fig2.add_vrect(x0=0,  x1=30,  fillcolor='#22c55e',
                       opacity=0.04, line_width=0)
        fig2.add_vrect(x0=70, x1=100, fillcolor='#ef4444',
                       opacity=0.04, line_width=0)
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title='Kurva Distribusi Nilai RSI — Seluruh Saham',
            xaxis_title='Nilai RSI', yaxis_title='Kerapatan',
            height=360, showlegend=False
        )
        fig2.update_xaxes(range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    with col_b:
        st.markdown("**Panduan RSI**")
        st.markdown("""
        <div class='box box-green'>
            <b>RSI &lt; 30 — Oversold</b><br>
            Harga terlalu murah, potensi rebound.<br>
            Pertimbangkan untuk <b>Beli</b>.
        </div>
        <div class='box box-gray'>
            <b>RSI 30 – 70 — Normal</b><br>
            Harga bergerak wajar.<br>
            Disarankan <b>Hold dan pantau</b>.
        </div>
        <div class='box box-red'>
            <b>RSI &gt; 70 — Overbought</b><br>
            Harga terlalu mahal, potensi koreksi.<br>
            Pertimbangkan untuk <b>Jual</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Statistik RSI**")
    st.dataframe(
        df_rsi['RSI'].describe().round(2).rename('Nilai'),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Distribusi Final Signal")

    signal_order = SIGNAL_ORDER
    signal_colors = SIGNAL_COLORS
    signal_counts = (df_saham['Final_Signal']
                     .value_counts()
                     .reindex(signal_order, fill_value=0)
                     .reset_index())
    signal_counts.columns = ['Sinyal', 'Jumlah']
    signal_counts['Persen'] = (
        signal_counts['Jumlah'] / signal_counts['Jumlah'].sum() * 100
    ).round(1)

    col_c, col_d = st.columns([3, 2])

    with col_c:
        fig3 = px.bar(
            signal_counts, x='Sinyal', y='Jumlah',
            color='Sinyal', color_discrete_map=signal_colors,
            text=signal_counts['Persen'].astype(str) + '%',
            title='Distribusi Final Signal — Seluruh Saham IDX'
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title='Jenis Sinyal', yaxis_title='Jumlah Data',
            showlegend=False, height=380
        )
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    with col_d:
        st.markdown("**Rangkuman Sinyal**")
        for _, row in signal_counts.iterrows():
            color = signal_colors.get(row['Sinyal'], '#94a3b8')
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        align-items:center; padding:10px 14px; margin:4px 0;
                        border-radius:8px; background:#f8fafc; border: 1px solid #e2e8f0;
                        border-left:3px solid {color}'>
                <span style='font-weight:500; font-size:0.85rem;
                             color:#374151'>{row['Sinyal']}</span>
                <span style='font-family:DM Mono,monospace; font-size:0.8rem;
                             color:#64748b'>{row['Jumlah']:,}
                    <span style='color:#94a3b8'> ({row['Persen']}%)</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("""
        <div class='box box-yellow' style='margin-top:12px'>
        Mayoritas sinyal adalah <b>Hold</b>, mencerminkan kondisi pasar 
        yang cenderung <i>sideways</i> pada periode ini.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Sumber data: IDX Stock Summary 2020–2024  "
        "·  Periode: Sep 2022 – Okt 2023  ·  CC26-PSU256"
    )

elif halaman == HALAMAN_SENTIMEN:

    st.title("Distribusi Sentimen Berita Saham")
    st.markdown(
        "Analisis sentimen dari **9.816 berita finansial** periode "
        "Januari 2024 – Maret 2025 yang digunakan sebagai konteks "
        "simulasi pada A/B Testing Grup B."
    )
    st.markdown("""
    <div class='box box-yellow'>
    <b>Catatan:</b> Sentimen berita yang ditampilkan merupakan simulasi 
    berdasarkan distribusi dataset berita finansial. Sentimen asli akan 
    digantikan dengan output model Deep Learning dari AI Engineer setelah 
    proses integrasi sistem selesai.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if not berita_loaded:
        st.error(
            "File dataset berita tidak ditemukan. "
            "Pastikan file CSV sudah tersedia di folder yang benar."
        )
    else:
        KOLOM_SENTIMEN = 'sentimen'
        KOLOM_JUDUL    = 'judul'
        KOLOM_TANGGAL  = 'tanggal'
        color_sent = SENTIMENT_COLORS

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Berita",
                  f"{len(df_berita):,}")
        c2.metric("Sentimen Positif",
                  f"{(df_berita[KOLOM_SENTIMEN]=='positif').sum():,}")
        c3.metric("Sentimen Negatif",
                  f"{(df_berita[KOLOM_SENTIMEN]=='negatif').sum():,}")

        st.markdown("---")

        col_e, col_f = st.columns(2)

        with col_e:
            st.subheader("Proporsi Sentimen")
            sent_counts = (df_berita[KOLOM_SENTIMEN]
                           .value_counts().reset_index())
            sent_counts.columns = ['Sentimen', 'Jumlah']
            fig4 = px.pie(
                sent_counts, names='Sentimen', values='Jumlah',
                color='Sentimen', color_discrete_map=color_sent
            )
            fig4.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hole=0.45,
                textfont_size=13
            )
            fig4.update_layout(
                **PLOTLY_LAYOUT,
                height=360, showlegend=True,
                legend=dict(orientation='h', yanchor='top',
                            y=-0.1, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig4, use_container_width=True,
                            config=PLOTLY_CONFIG)

        with col_f:
            st.subheader("Volume Berita per Bulan")
            if KOLOM_TANGGAL in df_berita.columns:
                df_berita[KOLOM_TANGGAL] = pd.to_datetime(
                    df_berita[KOLOM_TANGGAL], errors='coerce'
                )
                df_berita['Bulan'] = (df_berita[KOLOM_TANGGAL]
                                      .dt.to_period('M').astype(str))
                monthly = (df_berita
                           .groupby(['Bulan', KOLOM_SENTIMEN])
                           .size().reset_index(name='Jumlah'))
                fig5 = px.bar(
                    monthly, x='Bulan', y='Jumlah',
                    color=KOLOM_SENTIMEN,
                    color_discrete_map=color_sent,
                    barmode='stack'
                )
                fig5.update_layout(
                    **PLOTLY_LAYOUT,
                    xaxis_title='Bulan',
                    yaxis_title='Jumlah Berita',
                    height=360, xaxis_tickangle=45,
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig5, use_container_width=True,
                                config=PLOTLY_CONFIG)

        st.markdown("---")

        st.subheader("Contoh Berita per Sentimen")
        selected_sent = st.selectbox(
            "Filter Sentimen",
            options=['Semua', 'positif', 'netral', 'negatif'],
            key="filter_sentimen"
        )

        df_filtered = (
            df_berita if selected_sent == 'Semua'
            else df_berita[df_berita[KOLOM_SENTIMEN] == selected_sent]
        )
        cols_show = (
            [KOLOM_JUDUL, KOLOM_TANGGAL, KOLOM_SENTIMEN]
            if KOLOM_JUDUL in df_berita.columns
            else df_berita.columns.tolist()
        )
        st.dataframe(
            df_filtered[cols_show].head(5),
            use_container_width=True,
            hide_index=True
        )
        st.markdown(f"""
        <div class='box box-blue'>
        Menampilkan 5 contoh dari <b>{len(df_filtered):,}</b> berita 
        kategori <b>{selected_sent}</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption(
            "Sumber: Dataset-CNBCI-Sentimented  "
            "·  Periode: Jan 2024 – Mar 2025  ·  CC26-PSU256"
        )

elif halaman == HALAMAN_AB:

    st.title("Hasil A/B Testing")
    st.markdown(
        "Evaluasi komparasi performa dua pendekatan rekomendasi saham: "
        "**Grup A** menggunakan RSI + MA saja, dan **Grup B** menggunakan "
        "RSI + MA + Sentimen Berita."
    )
    st.markdown("""
    <div class='box box-blue'>
    <b>Tentang A/B Testing:</b> Metode membandingkan dua pendekatan 
    berbeda untuk membuktikan secara data mana yang lebih baik hasilnya,
    bukan berdasarkan asumsi semata.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    akurasi_a = 49.61
    akurasi_b = 49.28
    selisih   = akurasi_b - akurasi_a

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi Grup A",      f"{akurasi_a:.2f}%", "RSI + MA Only")
    c2.metric("Akurasi Grup B", f"{akurasi_b:.2f}%",
          delta=f"{selisih:+.2f}%",
          help="RSI + MA + Sentimen")
    c3.metric("Total Sinyal Grup A", f"{len(df_grup_a):,}")
    c4.metric("Total Sinyal Grup B", f"{len(df_grup_b):,}")

    st.markdown("---")

    st.subheader("Perbandingan Distribusi Sinyal")

    signal_order = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']

    cnt_a = (df_grup_a['Final_Signal']
             .value_counts()
             .reindex(signal_order, fill_value=0)
             .reset_index())
    cnt_a.columns = ['Sinyal', 'Jumlah']
    cnt_a['Grup'] = 'Grup A — RSI + MA'

    cnt_b = (df_grup_b['Signal_B']
             .value_counts()
             .reindex(signal_order, fill_value=0)
             .reset_index())
    cnt_b.columns = ['Sinyal', 'Jumlah']
    cnt_b['Grup'] = 'Grup B — RSI + MA + Sentimen'

    df_compare = pd.concat([cnt_a, cnt_b])
    fig6 = px.bar(
        df_compare, x='Sinyal', y='Jumlah',
        color='Grup', barmode='group',
        color_discrete_sequence=['#3b82f6', '#f59e0b'],
        text='Jumlah'
    )
    fig6.update_traces(
        texttemplate='%{text:,}', textposition='outside'
    )
    fig6.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title='Jenis Sinyal', yaxis_title='Jumlah Sinyal',
        height=420,
        legend=dict(orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig6, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    <div class='box box-blue'>
    Pergeseran distribusi sinyal antara Grup A dan Grup B menunjukkan 
    bahwa penambahan sentimen berita mempengaruhi rekomendasi yang dihasilkan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Hasil Uji Statistik")

    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown("**Grup A — Mann-Whitney U Test**")
        st.markdown(
            "Membandingkan akurasi sinyal RSI+MA terhadap "
            "baseline tebak-tebakan acak (50%)."
        )
        np.random.seed(42)
        baseline = np.random.binomial(1, 0.5, size=len(df_grup_a))
        u_stat, p_val_a = mannwhitneyu(
            df_grup_a['Prediksi_Benar'].values,
            baseline, alternative='greater'
        )
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0;
                    border-radius:10px; padding:20px; margin:8px 0'>
            <div style='display:flex; justify-content:space-between;
                        align-items:center; padding-bottom:12px;
                        margin-bottom:12px; border-bottom:1px solid #f1f5f9'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>U Statistic</span>
                <span style='font-family:DM Mono,monospace; font-size:1rem;
                             font-weight:500; color:#0f172a'>
                    {u_stat:,.2f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between;
                        align-items:center; padding-bottom:12px;
                        margin-bottom:12px; border-bottom:1px solid #f1f5f9'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>P-Value</span>
                <span style='font-family:DM Mono,monospace; font-size:1rem;
                             font-weight:500; color:#0f172a'>
                    {p_val_a:.4f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between;
                        align-items:center'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>Signifikan (α=0.05)</span>
                <span style='font-size:0.9rem; font-weight:600;
                             color:{"#166534" if p_val_a < 0.05 else "#991b1b"}'>
                    {"Ya" if p_val_a < 0.05 else "Tidak"}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if p_val_a < 0.05:
            st.markdown("""
            <div class='box box-green'>
            Sinyal RSI+MA terbukti lebih baik dari tebak-tebakan acak 
            secara statistik.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='box box-red'>
            Sinyal RSI+MA belum terbukti lebih baik dari tebak-tebakan acak secara statistik.
            </div>""", unsafe_allow_html=True)

    with col_h:
        st.markdown("**Grup B — Chi-Square Test**")
        st.markdown("Menguji apakah sentimen berita mengubah pola sinyal secara signifikan."
        )
        tabel    = pd.crosstab(df_grup_b['Signal_B'],
                               df_grup_b['Final_Signal'])
        chi2, p_val_b, dof, _ = chi2_contingency(tabel)
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0;
                    border-radius:10px; padding:20px; margin:8px 0'>
            <div style='display:flex; justify-content:space-between;
                        align-items:center; padding-bottom:12px;
                        margin-bottom:12px; border-bottom:1px solid #f1f5f9'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>Chi² Statistic</span>
                <span style='font-family:DM Mono,monospace; font-size:1rem;
                             font-weight:500; color:#0f172a'>
                    {chi2:,.4f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between;
                        align-items:center; padding-bottom:12px;
                        margin-bottom:12px; border-bottom:1px solid #f1f5f9'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>P-Value</span>
                <span style='font-family:DM Mono,monospace; font-size:1rem;
                             font-weight:500; color:#0f172a'>
                    {p_val_b:.4f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between;
                        align-items:center'>
                <span style='font-size:0.72rem; font-weight:600;
                             color:#94a3b8; text-transform:uppercase;
                             letter-spacing:0.06em'>Signifikan (α=0.05)</span>
                <span style='font-size:0.9rem; font-weight:600;
                             color:{"#166534" if p_val_b < 0.05 else "#991b1b"}'>
                    {"Ya" if p_val_b < 0.05 else "Tidak"}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if p_val_b < 0.05:
            st.markdown("""
            <div class='box box-green'>
            Penambahan sentimen berita terbukti mengubah pola sinyal 
            rekomendasi secara signifikan.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='box box-red'>
            Sentimen berita belum terbukti mengubah pola sinyal 
            secara signifikan.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Kesimpulan Akhir")
    st.markdown(f"""
    <div style='background:#ffffff; border:1px solid #e2e8f0;
                border-radius:12px; padding:28px'>
        <table style='width:100%; border-collapse:collapse;
                      font-size:0.875rem; margin-bottom:20px'>
            <thead>
                <tr style='border-bottom:2px solid #e2e8f0'>
                    <th style='text-align:left; padding:10px 12px;
                               color:#64748b; font-weight:600;
                               font-size:0.72rem; text-transform:uppercase;
                               letter-spacing:0.05em'>Metrik</th>
                    <th style='text-align:center; padding:10px 12px;
                               color:#3b82f6; font-weight:600;
                               font-size:0.72rem; text-transform:uppercase;
                               letter-spacing:0.05em'>Grup A</th>
                    <th style='text-align:center; padding:10px 12px;
                               color:#f59e0b; font-weight:600;
                               font-size:0.72rem; text-transform:uppercase;
                               letter-spacing:0.05em'>Grup B</th>
                </tr>
            </thead>
            <tbody>
                <tr style='border-bottom:1px solid #f1f5f9'>
                    <td style='padding:12px; color:#374151'>Pendekatan</td>
                    <td style='padding:12px; text-align:center;
                               color:#64748b'>RSI + MA</td>
                    <td style='padding:12px; text-align:center;
                               color:#64748b'>RSI + MA + Sentimen</td>
                </tr>
                <tr style='border-bottom:1px solid #f1f5f9'>
                    <td style='padding:12px; color:#374151'>Akurasi</td>
                    <td style='padding:12px; text-align:center;
                               font-family:DM Mono,monospace;
                               font-weight:500'>{akurasi_a:.2f}%</td>
                    <td style='padding:12px; text-align:center;
                               font-family:DM Mono,monospace;
                               font-weight:500'>{akurasi_b:.2f}%</td>
                </tr>
                <tr style='border-bottom:1px solid #f1f5f9'>
                    <td style='padding:12px; color:#374151'>Uji Statistik</td>
                    <td style='padding:12px; text-align:center;
                               color:#64748b'>Mann-Whitney U</td>
                    <td style='padding:12px; text-align:center;
                               color:#64748b'>Chi-Square</td>
                </tr>
                <tr>
                    <td style='padding:12px; color:#374151'>P-Value</td>
                    <td style='padding:12px; text-align:center;
                               font-family:DM Mono,monospace'>
                        {p_val_a:.4f}</td>
                    <td style='padding:12px; text-align:center;
                               font-family:DM Mono,monospace'>
                        {p_val_b:.4f}</td>
                </tr>
            </tbody>
        </table>
        <div style='font-size:0.875rem; color:#374151; line-height:1.8'>
            Sinyal RSI + MA menghasilkan akurasi <b>{akurasi_a:.2f}%</b>, 
            hampir setara dengan tebak-tebakan acak (50%). Penambahan sentimen 
            berita pada Grup B menghasilkan akurasi <b>{akurasi_b:.2f}%</b> 
            dengan selisih <b>{selisih:+.2f}%</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='box box-yellow' style='margin-top:12px'>
    <b>Catatan:</b> Sentimen yang digunakan masih berupa simulasi. 
    Hasil akan diperbarui setelah model Deep Learning dari AI Engineer 
    selesai diintegrasikan ke dalam sistem.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Coding Camp 2026 powered by DBS Foundation  ·  CC26-PSU256"
    )