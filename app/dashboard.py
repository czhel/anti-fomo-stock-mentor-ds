import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ─── KONFIGURASI HALAMAN ───────────────────────────────────────────
st.set_page_config(
    page_title="Anti FOMO Stock Mentor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font dan warna utama */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header utama */
    h1 { 
        font-family: 'Space Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #0f172a !important;
    }
    h2, h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Metric card */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.4rem !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem !important;
        padding: 8px 0 !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 24px 0;
    }

    /* Caption */
    .stCaption {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.875rem;
        color: #1e40af;
    }
    .warning-box {
        background: #fefce8;
        border-left: 4px solid #eab308;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.875rem;
        color: #854d0e;
    }
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.875rem;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────────────
@st.cache_data
def load_saham():
    df = pd.read_csv('../data/cleaned_stock_data.csv',
                     parse_dates=['Last Trading Date'])
    return df

@st.cache_data
def load_grup_a():
    df = pd.read_csv('../data/ab_testing_grup_a.csv',
                     parse_dates=['Last Trading Date'])
    return df

@st.cache_data
def load_grup_b():
    df = pd.read_csv('../data/ab_testing_grup_b.csv',
                     parse_dates=['Last Trading Date'])
    return df

@st.cache_data
def load_berita():
    df = pd.read_csv('../data/dataset_berita.csv')
    return df

# Load semua data
df_saham  = load_saham()
df_grup_a = load_grup_a()
df_grup_b = load_grup_b()

try:
    df_berita = load_berita()
    berita_loaded = True
except:
    berita_loaded = False

# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px'>
        <div style='font-size:2.5rem'>📈</div>
        <div style='font-family: Space Mono, monospace; font-size:0.95rem; 
                    font-weight:700; color:#f1f5f9; letter-spacing:0.03em'>
            Anti FOMO<br>Stock Mentor
        </div>
        <div style='font-size:0.7rem; color:#94a3b8; margin-top:4px'>
            Data Science Dashboard
        </div>
        <div style='font-size:0.65rem; color:#64748b; margin-top:2px'>
            CC26-PSU256
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    halaman = st.radio(
        "Navigasi Halaman",
        options=[
            "📊 EDA Dataset Saham",
            "📰 Distribusi Sentimen Berita",
            "🧪 Hasil A/B Testing"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569; text-align:center'>
        <div>Periode Data</div>
        <div style='font-family: Space Mono, monospace; color:#94a3b8'>
            Sep 2022 – Okt 2023
        </div>
        <br>
        <div>Coding Camp 2026</div>
        <div style='color:#64748b'>powered by DBS Foundation</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HALAMAN 1 — EDA DATASET SAHAM
# ══════════════════════════════════════════════════════════════════
if halaman == "📊 EDA Dataset Saham":

    st.title("📊 EDA Dataset Saham IDX")
    st.markdown("""
    Eksplorasi data saham IDX periode **September 2022 – Oktober 2023**
    yang digunakan sebagai dasar analisis teknikal pada proyek Anti FOMO Stock Mentor.
    """)
    st.markdown("---")

    # ── METRIC CARD ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Data",    f"{len(df_saham):,} baris")
    c2.metric("🏢 Jumlah Saham",  f"{df_saham['Stock Code'].nunique():,} saham")
    c3.metric("📅 Periode Awal",  str(df_saham['Last Trading Date'].min().date()))
    c4.metric("📅 Periode Akhir", str(df_saham['Last Trading Date'].max().date()))

    st.markdown("---")

    # ── GRAFIK 1: PERGERAKAN HARGA ────────────────────────────────
    st.subheader("📉 Pergerakan Harga Saham")

    stock_list     = sorted(df_saham['Stock Code'].unique().tolist())
    selected_stock = st.selectbox("Pilih Kode Saham", stock_list,
                                  index=stock_list.index('GOTO') 
                                  if 'GOTO' in stock_list else 0)

    df_stock = df_saham[df_saham['Stock Code'] == selected_stock].copy()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_stock['Last Trading Date'], y=df_stock['Close'],
        mode='lines', name='Harga Close',
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

    # Tandai sinyal Buy dan Sell
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
        title=f'Pergerakan Harga Saham {selected_stock} + Sinyal MA',
        xaxis_title='Tanggal', yaxis_title='Harga (IDR)',
        hovermode='x unified', height=420,
        plot_bgcolor='#fafafa', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig1.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    fig1.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(f"""
    <div class='info-box'>
    📌 Grafik menampilkan pergerakan harga penutupan saham <b>{selected_stock}</b> 
    beserta Moving Average 5 hari (MA5) dan 20 hari (MA20). 
    Segitiga hijau = sinyal beli, segitiga merah = sinyal jual.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── GRAFIK 2: DISTRIBUSI RSI ──────────────────────────────────
    st.subheader("📊 Distribusi RSI")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        df_rsi = df_saham.dropna(subset=['RSI'])
        fig2   = px.histogram(
            df_rsi, x='RSI', nbins=60,
            color_discrete_sequence=['#6366f1'],
            title='Distribusi Nilai RSI — Seluruh Saham IDX'
        )
        fig2.add_vline(x=30, line_dash='dash', line_color='#22c55e', line_width=2,
                       annotation_text='Oversold (30)')
        fig2.add_vline(x=70, line_dash='dash', line_color='#ef4444', line_width=2,
                       annotation_text='Overbought (70)')
        fig2.add_vrect(x0=0,  x1=30,  fillcolor='#22c55e', opacity=0.05, line_width=0)
        fig2.add_vrect(x0=70, x1=100, fillcolor='#ef4444', opacity=0.05, line_width=0)
        fig2.update_layout(
            xaxis_title='Nilai RSI', yaxis_title='Jumlah Data',
            height=350, plot_bgcolor='#fafafa', paper_bgcolor='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("**Cara Membaca RSI:**")
        st.markdown("""
        <div class='success-box'>
        🟢 <b>RSI &lt; 30</b><br>Harga terlalu murah<br>→ Pertimbangkan <b>Beli</b>
        </div>
        <div class='info-box'>
        ⚪ <b>RSI 30–70</b><br>Harga normal<br>→ <b>Tahan</b>
        </div>
        <div style='background:#fef2f2; border-left:4px solid #ef4444; 
                    border-radius:0 8px 8px 0; padding:12px 16px; 
                    margin:12px 0; font-size:0.875rem; color:#991b1b'>
        🔴 <b>RSI &gt; 70</b><br>Harga terlalu mahal<br>→ Pertimbangkan <b>Jual</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Statistik RSI:**")
        rsi_stat = df_rsi['RSI'].describe().round(2)
        st.dataframe(rsi_stat.rename('Nilai'), use_container_width=True)

    st.markdown("---")

    # ── GRAFIK 3: DISTRIBUSI SINYAL TEKNIKAL ─────────────────────
    st.subheader("🚦 Distribusi Final Signal")

    col_c, col_d = st.columns([2, 1])

    with col_c:
        signal_order  = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
        signal_colors = {
            'Strong Buy':  '#166534',
            'Buy':         '#22c55e',
            'Hold':        '#94a3b8',
            'Sell':        '#ef4444',
            'Strong Sell': '#7f1d1d'
        }
        signal_counts = (df_saham['Final_Signal']
                         .value_counts()
                         .reindex(signal_order, fill_value=0)
                         .reset_index())
        signal_counts.columns = ['Sinyal', 'Jumlah']
        signal_counts['Persen'] = (
            signal_counts['Jumlah'] / signal_counts['Jumlah'].sum() * 100
        ).round(1)

        fig3 = px.bar(
            signal_counts, x='Sinyal', y='Jumlah',
            color='Sinyal', color_discrete_map=signal_colors,
            text=signal_counts['Persen'].astype(str) + '%',
            title='Distribusi Final Signal — Seluruh Saham IDX'
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(
            xaxis_title='Jenis Sinyal', yaxis_title='Jumlah Sinyal',
            showlegend=False, height=400,
            plot_bgcolor='#fafafa', paper_bgcolor='white'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("**Keterangan Sinyal:**")
        for _, row in signal_counts.iterrows():
            color = signal_colors.get(row['Sinyal'], '#94a3b8')
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; 
                        padding:8px 12px; margin:4px 0; border-radius:8px;
                        background:#f8fafc; border-left:4px solid {color}'>
                <span style='font-weight:600; font-size:0.85rem'>{row['Sinyal']}</span>
                <span style='font-family: Space Mono, monospace; 
                             font-size:0.8rem; color:#64748b'>
                    {row['Jumlah']:,} ({row['Persen']}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='warning-box' style='margin-top:16px'>
        ⚠️ Mayoritas sinyal adalah <b>Hold</b>, mencerminkan kondisi 
        pasar yang cenderung <i>sideways</i> pada periode ini.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Sumber data: IDX Stock Summary 2020–2024 | Periode: Sep 2022 – Okt 2023 | Proyek: Anti FOMO Stock Mentor — CC26-PSU256")

# ══════════════════════════════════════════════════════════════════
# HALAMAN 2 — DISTRIBUSI SENTIMEN BERITA
# ══════════════════════════════════════════════════════════════════
elif halaman == "📰 Distribusi Sentimen Berita":

    st.title("📰 Distribusi Sentimen Berita Saham")
    st.markdown("""
    Analisis distribusi sentimen dari dataset berita finansial yang digunakan 
    sebagai konteks simulasi pada A/B Testing Grup B.
    """)
    st.markdown("""
    <div class='warning-box'>
    ⚠️ <b>Catatan:</b> Sentimen yang ditampilkan berasal dari dataset berita 
    finansial yang sudah di-cleaning. Pada A/B Testing, sentimen ini digunakan 
    sebagai simulasi karena model Deep Learning dari AI Engineer masih dalam 
    tahap pengembangan.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if not berita_loaded:
        st.error("""
        ❌ File dataset berita tidak ditemukan. 
        Pastikan file CSV dataset berita sudah ada di folder notebooks/ 
        dan nama file sudah sesuai di kode load_berita().
        """)
    else:
        # ── SESUAIKAN nama kolom berikut dengan dataset beritamu ──
        # Ganti 'sentimen' dengan nama kolom sentimen di datasetmu
        # Ganti 'judul' dengan nama kolom judul berita di datasetmu
        # Ganti 'tanggal' dengan nama kolom tanggal di datasetmu
        KOLOM_SENTIMEN = 'sentimen'   # ← ganti sesuai nama kolom
        KOLOM_JUDUL    = 'judul'      # ← ganti sesuai nama kolom
        KOLOM_TANGGAL  = 'tanggal'    # ← ganti sesuai nama kolom

        # ── METRIC CARD ──────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("📰 Total Berita", f"{len(df_berita):,}")
        c2.metric("😊 Sentimen Positif",
                  f"{(df_berita[KOLOM_SENTIMEN] == 'positif').sum():,}")
        c3.metric("😟 Sentimen Negatif",
                  f"{(df_berita[KOLOM_SENTIMEN] == 'negatif').sum():,}")

        st.markdown("---")

        col_e, col_f = st.columns([1, 1])

        # ── GRAFIK 4: PIE CHART DISTRIBUSI SENTIMEN ──────────────
        with col_e:
            st.subheader("🥧 Distribusi Sentimen")
            sent_counts = df_berita[KOLOM_SENTIMEN].value_counts().reset_index()
            sent_counts.columns = ['Sentimen', 'Jumlah']

            color_sent = {
                'positif': '#22c55e',
                'netral':  '#94a3b8',
                'negatif': '#ef4444'
            }

            fig4 = px.pie(
                sent_counts, names='Sentimen', values='Jumlah',
                color='Sentimen', color_discrete_map=color_sent,
                title='Proporsi Sentimen Berita'
            )
            fig4.update_traces(
                textposition='inside', textinfo='percent+label',
                hole=0.4
            )
            fig4.update_layout(height=380, paper_bgcolor='white')
            st.plotly_chart(fig4, use_container_width=True)

        # ── GRAFIK 5: BAR CHART VOLUME BERITA PER BULAN ──────────
        with col_f:
            st.subheader("📅 Volume Berita per Bulan")

            if KOLOM_TANGGAL in df_berita.columns:
                df_berita[KOLOM_TANGGAL] = pd.to_datetime(
                    df_berita[KOLOM_TANGGAL], errors='coerce'
                )
                df_berita['Bulan'] = df_berita[KOLOM_TANGGAL].dt.to_period('M').astype(str)

                monthly = (df_berita.groupby(['Bulan', KOLOM_SENTIMEN])
                           .size().reset_index(name='Jumlah'))

                fig5 = px.bar(
                    monthly, x='Bulan', y='Jumlah',
                    color=KOLOM_SENTIMEN,
                    color_discrete_map=color_sent,
                    title='Volume Berita per Bulan',
                    barmode='stack'
                )
                fig5.update_layout(
                    xaxis_title='Bulan', yaxis_title='Jumlah Berita',
                    height=380, plot_bgcolor='#fafafa',
                    paper_bgcolor='white',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Kolom tanggal tidak ditemukan di dataset berita.")

        st.markdown("---")

        # ── TABEL CONTOH BERITA PER SENTIMEN ─────────────────────
        st.subheader("🗞️ Contoh Berita per Sentimen")

        selected_sent = st.selectbox(
            "Pilih Jenis Sentimen",
            options=['positif', 'netral', 'negatif']
        )

        df_filtered = df_berita[
            df_berita[KOLOM_SENTIMEN] == selected_sent
        ]

        if KOLOM_JUDUL in df_berita.columns:
            st.dataframe(
                df_filtered[[KOLOM_JUDUL, KOLOM_SENTIMEN]].head(5),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(
                df_filtered.head(5),
                use_container_width=True,
                hide_index=True
            )

        st.markdown(f"""
        <div class='info-box'>
        📌 Menampilkan 5 contoh berita dengan sentimen <b>{selected_sent}</b> 
        dari total {len(df_filtered):,} berita.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Sumber: Dataset Berita Finansial | Periode: Januari 2024 – Maret 2025")

# ══════════════════════════════════════════════════════════════════
# HALAMAN 3 — HASIL A/B TESTING
# ══════════════════════════════════════════════════════════════════
elif halaman == "🧪 Hasil A/B Testing":

    st.title("🧪 Hasil A/B Testing")
    st.markdown("""
    Perbandingan akurasi dua pendekatan rekomendasi saham:
    **Grup A** menggunakan RSI + MA saja, dan **Grup B** menggunakan 
    RSI + MA + Sentimen Berita.
    """)
    st.markdown("""
    <div class='info-box'>
    💡 <b>Apa itu A/B Testing?</b> Metode membandingkan dua pendekatan berbeda 
    untuk membuktikan secara data mana yang lebih baik hasilnya.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── HITUNG AKURASI ────────────────────────────────────────────
    akurasi_a = df_grup_a['Prediksi_Benar'].mean() * 100
    akurasi_b = df_grup_b['Prediksi_Benar_B'].mean() * 100
    selisih   = akurasi_b - akurasi_a

    # ── METRIC CARD ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🅰️ Akurasi Grup A",
              f"{akurasi_a:.2f}%",
              "RSI + MA Only")
    c2.metric("🅱️ Akurasi Grup B",
              f"{akurasi_b:.2f}%",
              "RSI + MA + Sentimen",
              delta=f"{selisih:+.2f}%")
    c3.metric("📊 Total Sinyal Grup A", f"{len(df_grup_a):,}")
    c4.metric("📊 Total Sinyal Grup B", f"{len(df_grup_b):,}")

    st.markdown("---")

    # ── GRAFIK 6: GROUPED BAR PERBANDINGAN SINYAL ─────────────────
    st.subheader("📊 Perbandingan Distribusi Sinyal Grup A vs Grup B")

    signal_order  = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
    signal_colors = {
        'Strong Buy':  '#166534',
        'Buy':         '#22c55e',
        'Hold':        '#94a3b8',
        'Sell':        '#ef4444',
        'Strong Sell': '#7f1d1d'
    }

    cnt_a = (df_grup_a['Final_Signal']
             .value_counts()
             .reindex(signal_order, fill_value=0)
             .reset_index())
    cnt_a.columns = ['Sinyal', 'Jumlah']
    cnt_a['Grup'] = 'Grup A (RSI + MA)'

    cnt_b = (df_grup_b['Signal_B']
             .value_counts()
             .reindex(signal_order, fill_value=0)
             .reset_index())
    cnt_b.columns = ['Sinyal', 'Jumlah']
    cnt_b['Grup'] = 'Grup B (RSI + MA + Sentimen)'

    df_compare = pd.concat([cnt_a, cnt_b])

    fig6 = px.bar(
        df_compare, x='Sinyal', y='Jumlah',
        color='Grup', barmode='group',
        color_discrete_sequence=['#3b82f6', '#f59e0b'],
        title='Perbandingan Distribusi Sinyal Grup A vs Grup B',
        text='Jumlah'
    )
    fig6.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig6.update_layout(
        xaxis_title='Jenis Sinyal', yaxis_title='Jumlah Sinyal',
        height=420, plot_bgcolor='#fafafa', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1)
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    📌 Grafik menampilkan perubahan distribusi sinyal setelah ditambahkan 
    sentimen berita. Pergeseran pada sinyal Hold, Buy, dan Sell menunjukkan 
    bahwa sentimen berita mempengaruhi rekomendasi yang dihasilkan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── UJI STATISTIK ─────────────────────────────────────────────
    st.subheader("📐 Hasil Uji Statistik")

    col_g, col_h = st.columns(2)

    # Grup A — Mann-Whitney
    with col_g:
        st.markdown("**🅰️ Grup A — Mann-Whitney U Test**")
        st.markdown("*Membandingkan akurasi RSI+MA vs baseline random 50%*")

        baseline = np.random.binomial(1, 0.5, size=len(df_grup_a))
        u_stat, p_val_a = mannwhitneyu(
            df_grup_a['Prediksi_Benar'].values,
            baseline,
            alternative='greater'
        )

        st.markdown(f"""
        <div style='background:#f8fafc; border-radius:12px; padding:20px; 
                    border:1px solid #e2e8f0; margin:8px 0'>
            <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    U STATISTIC
                </span>
                <span style='font-family:Space Mono,monospace; font-size:1.1rem; 
                             font-weight:700; color:#0f172a'>
                    {u_stat:,.2f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    P-VALUE
                </span>
                <span style='font-family:Space Mono,monospace; font-size:1.1rem; 
                             font-weight:700; color:#0f172a'>
                    {p_val_a:.4f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    SIGNIFIKAN (α=0.05)
                </span>
                <span style='font-weight:700; color:{"#166534" if p_val_a < 0.05 else "#991b1b"}'>
                    {"✅ Ya" if p_val_a < 0.05 else "❌ Tidak"}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if p_val_a < 0.05:
            st.markdown("""
            <div class='success-box'>
            ✅ Sinyal RSI+MA terbukti lebih baik dari tebak-tebakan acak 
            secara statistik.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#fef2f2; border-left:4px solid #ef4444; 
                        border-radius:0 8px 8px 0; padding:12px 16px; 
                        font-size:0.875rem; color:#991b1b'>
            ❌ Sinyal RSI+MA belum terbukti lebih baik dari tebak-tebakan. 
            Ini memperkuat alasan mengapa sentimen berita dibutuhkan.
            </div>
            """, unsafe_allow_html=True)

    # Grup B — Chi-Square
    with col_h:
        st.markdown("**🅱️ Grup B — Chi-Square Test**")
        st.markdown("*Menguji apakah sentimen mengubah pola sinyal rekomendasi*")

        tabel = pd.crosstab(df_grup_b['Signal_B'], df_grup_b['Prediksi_Benar_B'])
        chi2, p_val_b, dof, _ = chi2_contingency(tabel)

        st.markdown(f"""
        <div style='background:#f8fafc; border-radius:12px; padding:20px; 
                    border:1px solid #e2e8f0; margin:8px 0'>
            <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    CHI² STATISTIC
                </span>
                <span style='font-family:Space Mono,monospace; font-size:1.1rem; 
                             font-weight:700; color:#0f172a'>
                    {chi2:,.2f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:12px'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    P-VALUE
                </span>
                <span style='font-family:Space Mono,monospace; font-size:1.1rem; 
                             font-weight:700; color:#0f172a'>
                    {p_val_b:.4f}
                </span>
            </div>
            <div style='display:flex; justify-content:space-between'>
                <span style='color:#64748b; font-size:0.8rem; font-weight:600'>
                    SIGNIFIKAN (α=0.05)
                </span>
                <span style='font-weight:700; color:{"#166534" if p_val_b < 0.05 else "#991b1b"}'>
                    {"✅ Ya" if p_val_b < 0.05 else "❌ Tidak"}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if p_val_b < 0.05:
            st.markdown("""
            <div class='success-box'>
            ✅ Penambahan sentimen berita terbukti mengubah pola sinyal 
            rekomendasi secara signifikan.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#fef2f2; border-left:4px solid #ef4444; 
                        border-radius:0 8px 8px 0; padding:12px 16px; 
                        font-size:0.875rem; color:#991b1b'>
            ❌ Sentimen berita belum terbukti mengubah pola sinyal 
            secara signifikan.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KESIMPULAN AKHIR ──────────────────────────────────────────
    st.subheader("📝 Kesimpulan Akhir")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg, #f8fafc, #eff6ff); 
                border-radius:16px; padding:28px; border:1px solid #dbeafe'>
        <h4 style='color:#1e40af; margin-top:0; font-family:Space Mono,monospace'>
            Hasil A/B Testing — Anti FOMO Stock Mentor
        </h4>
        <table style='width:100%; border-collapse:collapse; margin:16px 0'>
            <tr style='background:#dbeafe'>
                <th style='padding:10px; text-align:left; font-size:0.8rem; color:#1e40af'>
                    Metrik
                </th>
                <th style='padding:10px; text-align:center; font-size:0.8rem; color:#1e40af'>
                    Grup A (RSI + MA)
                </th>
                <th style='padding:10px; text-align:center; font-size:0.8rem; color:#1e40af'>
                    Grup B (RSI + MA + Sentimen)
                </th>
            </tr>
            <tr style='background:white'>
                <td style='padding:10px; font-size:0.85rem'>Akurasi</td>
                <td style='padding:10px; text-align:center; font-family:Space Mono,monospace; font-weight:700'>
                    {akurasi_a:.2f}%
                </td>
                <td style='padding:10px; text-align:center; font-family:Space Mono,monospace; font-weight:700'>
                    {akurasi_b:.2f}%
                </td>
            </tr>
            <tr style='background:#f8fafc'>
                <td style='padding:10px; font-size:0.85rem'>Uji Statistik</td>
                <td style='padding:10px; text-align:center; font-size:0.85rem'>Mann-Whitney U</td>
                <td style='padding:10px; text-align:center; font-size:0.85rem'>Chi-Square</td>
            </tr>
            <tr style='background:white'>
                <td style='padding:10px; font-size:0.85rem'>P-Value</td>
                <td style='padding:10px; text-align:center; font-family:Space Mono,monospace'>
                    {p_val_a:.4f}
                </td>
                <td style='padding:10px; text-align:center; font-family:Space Mono,monospace'>
                    {p_val_b:.4f}
                </td>
            </tr>
        </table>
        <div style='margin-top:16px; font-size:0.875rem; color:#374151; line-height:1.7'>
            Berdasarkan hasil A/B Testing, sinyal RSI + MA saja menghasilkan akurasi 
            sebesar <b>{akurasi_a:.2f}%</b> yang hampir sama dengan tebak-tebakan acak (50%), 
            membuktikan bahwa indikator teknikal saja tidak cukup. Penambahan sentimen berita 
            pada Grup B menghasilkan akurasi <b>{akurasi_b:.2f}%</b> dengan selisih 
            <b>{selisih:+.2f}%</b>.
        </div>
        <div style='margin-top:12px; padding:12px; background:#fefce8; border-radius:8px;
                    font-size:0.8rem; color:#854d0e'>
        ⚠️ <b>Catatan:</b> Sentimen yang digunakan pada pengujian ini masih berupa simulasi 
        berdasarkan distribusi dataset berita, bukan output model Deep Learning yang 
        sesungguhnya. Hasil akan diperbarui setelah model AI Engineer selesai diintegrasikan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Coding Camp 2026 powered by DBS Foundation | CC26-PSU256")