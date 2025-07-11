import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('toyota_model.sav', 'rb'))

# Judul aplikasi
st.set_page_config(page_title="Prediksi Kategori Penjualan Toyota", layout="centered")
st.title("🚗 Prediksi Kategori Penjualan Mobil Toyota")
st.markdown("Masukkan data penjualan untuk memprediksi apakah termasuk kategori **Rendah**, **Sedang**, atau **Tinggi**.")

# Input pengguna
model_mobil = st.selectbox("Model Mobil", ['Avanza', 'Yaris', 'Corolla', 'Camry', 'RAV4', 'Hilux', 'Innova', 'Fortuner'])
wilayah = st.selectbox("Wilayah Penjualan", ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar'])
pendapatan = st.number_input("Pendapatan (Rp)", min_value=0)

# Konversi input ke dataframe
if st.button("Prediksi"):
    # One-hot encoding manual
    input_df = pd.DataFrame({
        'Pendapatan': [pendapatan],
        'Model_' + model_mobil: [1],
        'Wilayah_' + wilayah: [1]
    })

    # Tambahkan kolom one-hot yang tidak dipilih supaya sesuai dengan training
    model_columns = ['Pendapatan'] + \
        ['Model_Avanza', 'Model_Yaris', 'Model_Corolla', 'Model_Camry', 'Model_RAV4', 'Model_Hilux', 'Model_Innova', 'Model_Fortuner'] + \
        ['Wilayah_Jakarta', 'Wilayah_Bandung', 'Wilayah_Surabaya', 'Wilayah_Medan', 'Wilayah_Makassar']
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Prediksi
    prediction = model.predict(input_df)[0]
    st.subheader("📊 Hasil Prediksi:")
    st.success(f"Kategori Penjualan: **{prediction}**")
