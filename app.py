import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# Judul
st.title("Prediksi Menggunakan Naive Bayes")

# Cek apakah model tersedia
model_path = 'model/naive_bayes_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model tidak ditemukan. Pastikan file 'naive_bayes_model.pkl' ada di folder 'model'.")
    st.stop()

# Batas input manual
limits = {
    "Pregnancies": (0, 20),
    "Glucose": (50, 200),
    "Blood Pressure": (40, 140),
    "Skin Thickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (10, 70),
    "Diabetes Pedigree Function": (0.0, 2.5),
    "Age": (10, 100)
}

# Tampilkan batas minimum dan maksimum (tanpa koma di belakang angka)
with st.expander("Lihat Batas Minimum dan Maksimum"):
    df_limits = pd.DataFrame({
        "Min": [f"{int(v[0])}" if float(v[0]).is_integer() else f"{v[0]:.2f}" for v in limits.values()],
        "Max": [f"{int(v[1])}" if float(v[1]).is_integer() else f"{v[1]:.2f}" for v in limits.values()]
    }, index=limits.keys())
    st.table(df_limits)

# Form input
with st.form("input_form"):
    pregnancies = st.number_input("Pregnancies", min_value=limits["Pregnancies"][0], max_value=limits["Pregnancies"][1], value=1, step=1)
    glucose = st.number_input("Glucose", min_value=limits["Glucose"][0], max_value=limits["Glucose"][1], value=100, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=limits["Blood Pressure"][0], max_value=limits["Blood Pressure"][1], value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=limits["Skin Thickness"][0], max_value=limits["Skin Thickness"][1], value=20, step=1)
    insulin = st.number_input("Insulin", min_value=limits["Insulin"][0], max_value=limits["Insulin"][1], value=80, step=1)
    bmi = st.number_input("BMI", min_value=float(limits["BMI"][0]), max_value=float(limits["BMI"][1]), value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=limits["Diabetes Pedigree Function"][0], max_value=limits["Diabetes Pedigree Function"][1], value=0.5, step=0.01, help="Skor riwayat diabetes keluarga")
    age = st.number_input("Age", min_value=limits["Age"][0], max_value=limits["Age"][1], value=30, step=1)

    submitted = st.form_submit_button("Prediksi")

# Proses prediksi
if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Pasien Diprediksi MENGIDAP Diabetes.")
    else:
        st.success("✅ Pasien Diprediksi TIDAK Mengidap Diabetes.")

    # Tampilkan probabilitas prediksi jika tersedia
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
        st.info(f"Probabilitas Diabetes: {prob:.2%}")
