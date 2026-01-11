import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Diagnosa COVID-19",
    layout="centered"
)

def load_model():
    try:
        model_data = joblib.load('model_decision_tree_covid19.pkl')
        return model_data
    except:
        return None

def load_data():
    try:
        df = pd.read_csv('data_gejala_covid19.csv')
        return df
    except:
        return None

model_data = load_model()
df = load_data()

st.title("Sistem Diagnosa COVID-19")
st.write("Menggunakan Decision Tree untuk analisis gejala")

# Informasi mahasiswa di sidebar
st.sidebar.write("**IDENTITAS MAHASISWA**")
st.sidebar.write("Nama: Haris Ramdhani")
st.sidebar.write("NPM: 20241310033")
st.sidebar.write("Prodi: Teknik Informatika")
st.sidebar.write("---")

# Tombol navigasi di beranda
st.write("### Menu Navigasi")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(" Diagnosa"):
        st.session_state.page = "diagnosa"
        st.rerun()

with col2:
    if st.button(" Data"):
        st.session_state.page = "data"
        st.rerun()

with col3:
    if st.button(" Model"):
        st.session_state.page = "model"
        st.rerun()

with col4:
    if st.button(" Tentang"):
        st.session_state.page = "tentang"
        st.rerun()

# Inisialisasi session state untuk halaman
if 'page' not in st.session_state:
    st.session_state.page = "beranda"

# Konten berdasarkan halaman yang dipilih
if st.session_state.page == "beranda":
    st.write("---")
    st.subheader("Selamat Datang di Sistem Diagnosa COVID-19")
    
    # Informasi tugas
    st.write("### Tentang Tugas Ini")
    st.write("Website ini dibuat untuk memenuhi tugas Ujian Akhir Semester (UAS) mata kuliah **Struktur Data** yang diampu oleh:")
    st.write("**Bapak Deni Suprihadi, S.T, M.KOM., MCE.**")
    
    st.write("### Deskripsi Sistem")
    st.write("Sistem ini menggunakan algoritma **Decision Tree** untuk menganalisis gejala COVID-19 dan memberikan prediksi berdasarkan input pengguna.")
    
    st.write("### Fitur Utama:")
    st.write("1. **Diagnosa** - Input gejala dan dapatkan hasil prediksi")
    st.write("2. **Data** - Lihat dataset gejala COVID-19 yang digunakan")
    st.write("3. **Model** - Lihat visualisasi decision tree")
    st.write("4. **Tentang** - Informasi lengkap tentang sistem")
    
    st.write("### Gejala yang Dianalisis:")
    st.write("- Demam (Tinggi/Sedang/Rendah)")
    st.write("- Batuk (Parah/Ringan/Tidak)")
    st.write("- Sesak Nafas (Ya/Tidak)")
    st.write("- Sakit Tenggorokan (Ya/Tidak)")
    st.write("- Kehilangan Rasa/Penciuman (Ya/Tidak)")
    
    st.write("### Cara Menggunakan:")
    st.write("1. Pilih menu **Diagnosa**")
    st.write("2. Masukkan gejala yang dialami")
    st.write("3. Klik tombol **Lakukan Diagnosa**")
    st.write("4. Lihat hasil dan rekomendasi")

elif st.session_state.page == "diagnosa":
    st.write("---")
    st.subheader("Diagnosa COVID-19")
    
    if model_data is None:
        st.warning("Model tidak tersedia. Jalankan Soal 2 terlebih dahulu.")
    else:
        clf = model_data['model']
        le_dict = model_data['encoders']
        feature_names = model_data['feature_names']
        
        # Form input gejala
        with st.form("form_diagnosa"):
            st.write("Silakan masukkan gejala yang dialami:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                demam = st.selectbox("Demam", ["Tinggi", "Sedang", "Rendah"], 
                                   help="Tinggi (>38°C), Sedang (37.5-38°C), Rendah (<37.5°C)")
                batuk = st.selectbox("Batuk", ["Parah", "Ringan", "Tidak"],
                                   help="Parah (terus-menerus), Ringan (sesekali), Tidak (tidak ada)")
                sesak_nafas = st.selectbox("Sesak Nafas", ["Ya", "Tidak"],
                                         help="Ya (sulit bernafas), Tidak (nafas normal)")
            
            with col2:
                sakit_tenggorokan = st.selectbox("Sakit Tenggorokan", ["Ya", "Tidak"],
                                               help="Ya (nyeri saat menelan), Tidak (normal)")
                kehilangan_rasa = st.selectbox("Kehilangan Rasa/Penciuman", ["Ya", "Tidak"],
                                             help="Ya (hilang penciuman/pengecap), Tidak (normal)")
            
            submitted = st.form_submit_button("Lakukan Diagnosa")
            
            if submitted:
                input_values = {
                    'demam': demam,
                    'batuk': batuk,
                    'sesak_nafas': sesak_nafas,
                    'sakit_tenggorokan': sakit_tenggorokan,
                    'kehilangan_rasa': kehilangan_rasa
                }
                
                # Konversi input ke numerik
                input_encoded = []
                for feature in feature_names:
                    if feature in input_values:
                        value = input_values[feature]
                        encoded_value = le_dict[feature].transform([value])[0]
                        input_encoded.append(encoded_value)
                    else:
                        input_encoded.append(0)
                
                # Prediksi
                input_array = np.array(input_encoded).reshape(1, -1)
                prediksi = clf.predict(input_array)[0]
                proba = clf.predict_proba(input_array)[0]
                
                st.write("---")
                st.write("### Hasil Diagnosa:")
                
                # Tampilkan gejala yang dimasukkan
                st.write("**Gejala yang dimasukkan:**")
                for gejala, nilai in input_values.items():
                    st.write(f"- {gejala.replace('_', ' ').title()}: {nilai}")
                
                st.write("---")
                
                # Tampilkan hasil
                if prediksi == 1:
                    st.error(f"**POSITIF COVID-19**")
                    st.write(f"**Probabilitas:** {proba[1]*100:.1f}%")
                    st.write("**Rekomendasi:**")
                    st.write("1. Segera lakukan tes PCR untuk konfirmasi")
                    st.write("2. Lakukan isolasi mandiri minimal 5 hari")
                    st.write("3. Gunakan masker dan jaga jarak")
                    st.write("4. Hubungi layanan kesehatan terdekat")
                    st.write("5. Pantau saturasi oksigen secara rutin")
                else:
                    st.success(f"**NEGATIF COVID-19**")
                    st.write(f"**Probabilitas:** {proba[0]*100:.1f}%")
                    st.write("**Rekomendasi:**")
                    st.write("1. Tetap jaga protokol kesehatan")
                    st.write("2. Lanjutkan aktivitas dengan hati-hati")
                    st.write("3. Monitor gejala secara berkala")
                    st.write("4. Jaga daya tahan tubuh")
                    st.write("5. Lakukan tes jika muncul gejala baru")
    
    if st.button("Kembali ke Beranda"):
        st.session_state.page = "beranda"
        st.rerun()

elif st.session_state.page == "data":
    st.write("---")
    st.subheader("Data Gejala COVID-19")
    
    if df is None:
        st.warning("Data tidak tersedia. Jalankan Soal 1 terlebih dahulu.")
    else:
        st.write(f"**Total Data:** {len(df)} pasien")
        
        # Tampilkan data
        st.write("### Tabel Data Gejala:")
        st.dataframe(df, use_container_width=True)
        
        # Statistik
        st.write("### Statistik Data:")
        col1, col2 = st.columns(2)
        with col1:
            positif = len(df[df['covid_positif'] == 'Ya'])
            st.metric("Positif COVID-19", positif)
        with col2:
            negatif = len(df[df['covid_positif'] == 'Tidak'])
            st.metric("Negatif COVID-19", negatif)
        
        # Distribusi gejala
        st.write("### Distribusi Gejala:")
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Demam:**")
            st.write(df['demam'].value_counts())
        with col4:
            st.write("**Batuk:**")
            st.write(df['batuk'].value_counts())
        
        col5, col6 = st.columns(2)
        with col5:
            st.write("**Sesak Nafas:**")
            st.write(df['sesak_nafas'].value_counts())
        with col6:
            st.write("**Kehilangan Rasa:**")
            st.write(df['kehilangan_rasa'].value_counts())
    
    if st.button("Kembali ke Beranda"):
        st.session_state.page = "beranda"
        st.rerun()

elif st.session_state.page == "model":
    st.write("---")
    st.subheader("Model Decision Tree")
    
    if model_data is None:
        st.warning("Model tidak tersedia. Jalankan Soal 2 terlebih dahulu.")
    else:
        clf = model_data['model']
        
        # Tampilkan diagram
        st.write("### Diagram Decision Tree:")
        try:
            image = Image.open('decision_tree_covid19_detailed.png')
            st.image(image, caption="Diagram Decision Tree untuk Diagnosa COVID-19", use_container_width=True)
        except:
            st.write("Diagram tidak ditemukan. Pastikan file 'diagram_decision_tree_covid19_detailed.png' ada di folder yang sama.")
        
        # Feature importance
        st.write("### Feature Importance:")
        st.write("Kontribusi setiap fitur dalam pengambilan keputusan:")
        
        feature_importance = pd.DataFrame({
            'Gejala': model_data['feature_names'],
            'Importance': clf.feature_importances_
        })
        
        for idx, row in feature_importance.iterrows():
            st.write(f"- **{row['Gejala'].replace('_', ' ').title()}**: {row['Importance']:.3f} ({row['Importance']*100:.1f}%)")
    
    if st.button("Kembali ke Beranda"):
        st.session_state.page = "beranda"
        st.rerun()

elif st.session_state.page == "tentang":
    st.write("---")
    st.subheader("Tentang Sistem")
    
    st.write("### Informasi Sistem:")
    st.write("**Nama Sistem:** Sistem Diagnosa COVID-19 menggunakan Decision Tree")
    st.write("**Tujuan:** Skrining awal gejala COVID-19 berdasarkan algoritma machine learning")
    st.write("**Metode:** Algoritma Decision Tree Classifier")
    st.write("**Criterion:** Entropy")
    st.write("**Data Training:** 25 sampel data gejala")
    
    st.write("### Informasi Akademik:")
    st.write("**Mata Kuliah:** Struktur Data")
    st.write("**Program Studi:** Teknik Informatika")
    st.write("**Dosen Pengampu:** Deni Suprihadi, S.T, M.KOM., MCE.")
    st.write("**Universitas:** Kebangsaan Republik Indonesia")
    st.write("**Tahun Akademik:** 2025/2026")
    
    st.write("### Identitas Pembuat:")
    st.write("**Nama:** Haris Ramdhani")
    st.write("**NPM:** 20241310033")
    st.write("**Prodi:** Teknik Informatika")
    
    st.write("### Teknologi yang Digunakan:")
    st.write("- Python 3.x")
    st.write("- Streamlit (Framework Web)")
    st.write("- Scikit-learn (Machine Learning)")
    st.write("- Pandas (Data Manipulation)")
    st.write("- Joblib (Model Serialization)")
    
    st.write("### Catatan Penting:")
    st.write("1. Sistem ini dibuat untuk tujuan edukasi dan penelitian")
    st.write("2. Hasil diagnosa **bukan** pengganti pemeriksaan medis")
    st.write("3. Konsultasikan dengan dokter untuk diagnosis yang akurat")
    st.write("4. Data pelatihan terbatas (25 sampel)")
    st.write("5. Model perlu di-training dengan data lebih banyak untuk aplikasi nyata")
    
    if st.button("Kembali ke Beranda"):
        st.session_state.page = "beranda"
        st.rerun()

st.write("---")
st.caption("UAS Struktur Data - Teknik Informatika - Universitas Kebangsaan Republik Indonesia")
st.caption("© 2025 - Haris Ramdhani (20241310033)")