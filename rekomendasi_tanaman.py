import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import plotly.express as px

# --- Header & config ---
st.set_page_config(page_title="🌱 Rekomendasi Tanaman", layout="wide")
st.markdown("""
<div style='background: linear-gradient(90deg,#2a9d8f,#26a98c); padding: 18px; border-radius: 10px;'>
  <h1 style='color: white; margin: 0;'>🌱 Aplikasi Rekomendasi Tanaman</h1>
  <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0;'>Rekomendasi tanaman berdasarkan kondisi lingkungan (otomatis memuat dataset lokal)</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
#### 📊 **Metode Machine Learning**
- **Algoritma**: Random Forest Classifier (ensemble dari 150 decision trees)
- **Fitur Masukan**: Suhu, Kelembaban, pH Tanah, Ketersediaan Air, Musim
- **Pembagian Data**: 80% training, 20% testing
- Random Forest menggabungkan banyak pohon keputusan untuk prediksi yang akurat dan stabil
""")
st.markdown("---")

# session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False


def load_and_train():
    """Load dataset from CSV, preprocess, train model and store in session_state."""
    try:
        df = pd.read_csv('rekomendasi_tanaman.csv')
    except FileNotFoundError:
        st.error("File 'rekomendasi_tanaman.csv' tidak ditemukan di folder aplikasi.")
        return

    with st.spinner('Memuat & menyiapkan dataset...'):
        time.sleep(0.6)
        # keep original musim for display
        df['musim_raw'] = df['musim'].astype(str)
        le = LabelEncoder()
        df['musim'] = le.fit_transform(df['musim_raw'])

        X = df[['suhu', 'kelembaban', 'ph_tanah', 'ketersediaan_air', 'musim']]
        y = df['tanaman']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    # store in session state
    st.session_state.data = df
    st.session_state.le_musim = le
    st.session_state.model = model
    st.session_state.accuracy = acc
    st.session_state.data_loaded = True
    st.session_state.model_ready = True


# Sidebar: controls
with st.sidebar:
    st.header('Kontrol Aplikasi')
    if not st.session_state.data_loaded:
        if st.button('🔄 Load Dataset & Latih Model'):
            load_and_train()
        st.write('Dataset akan dimuat dari `rekomendasi_tanaman.csv` di folder proyek.')
    else:
        st.success('✅ Dataset ter-load dan model siap')
        st.write(f'Data rows: {len(st.session_state.data)}')
        st.write(f'Model accuracy: {st.session_state.accuracy:.2%}')
    st.markdown('---')


# Main
if st.session_state.data_loaded:
    st.subheader('Preview Dataset (5 baris)')
    st.dataframe(st.session_state.data.head(5))

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader('Distribusi Tanaman')
        vc = st.session_state.data['tanaman'].value_counts().reset_index()
        vc.columns = ['tanaman', 'count']
        fig = px.bar(vc, x='tanaman', y='count', color='tanaman', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader('Statistik Singkat')
        st.metric('Total Sampel', len(st.session_state.data))
        st.metric('Varian Musim', st.session_state.data['musim_raw'].nunique())
        st.metric('Akurasi Model', f"{st.session_state.accuracy:.1%}")

    st.markdown('---')
    st.subheader('Input untuk Prediksi')

    ic1, ic2 = st.columns(2)
    with ic1:
        suhu = st.slider('Suhu (°C)', 0.0, 50.0, 25.0)
        kelembaban = st.slider('Kelembaban (%)', 0.0, 100.0, 75.0)
    with ic2:
        ph_tanah = st.slider('pH Tanah', 0.0, 14.0, 6.5)
        ketersediaan_air = st.slider('Ketersediaan Air (mm)', 0.0, 500.0, 200.0)
    with ic1:
        musim_input = st.selectbox('Musim', options=list(st.session_state.le_musim.classes_))
        
    with ic1:
        predict_btn = st.button('🔮 Prediksi Tanaman', disabled=not st.session_state.model_ready)

    if predict_btn:
        musim_encoded = st.session_state.le_musim.transform([musim_input])[0]
        input_df = pd.DataFrame({
            'suhu':[suhu],
            'kelembaban':[kelembaban],
            'ph_tanah':[ph_tanah],
            'ketersediaan_air':[ketersediaan_air],
            'musim':[musim_encoded]
        })
        with st.spinner('Memprediksi...'):
            time.sleep(0.8)
            pred = st.session_state.model.predict(input_df)[0]
        st.success(f'Rekomendasi tanaman: **{pred}**')
        if pred == 'semangka':
            st.info('Cocok untuk lahan terbuka; pastikan irigasi cukup dan struktur tanah gembur.')
        elif pred == 'padi':
            st.info('Butuh lahan tergenang/irigasi terkontrol; cocok untuk musim hujan.')
        elif pred == 'jagung':
            st.info('Tahan kemarau; cocok untuk tanah dengan drainase baik.')

else:
    st.info('Tekan tombol "Load Dataset & Latih Model" di sidebar untuk memulai. Aplikasi otomatis membaca `rekomendasi_tanaman.csv` dari folder proyek.')

st.markdown('---')
st.caption('Catatan: Model sederhana untuk demonstrasi; gunakan dataset besar dan validasi tambahan untuk produksi.')