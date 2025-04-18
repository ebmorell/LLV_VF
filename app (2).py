import streamlit as st
import joblib
import numpy as np

# Load trained LightGBM model
model = joblib.load("modelo_llv_lightgbm.pkl")

st.title("Virological Failure Prediction in Patients with Low-Level Viremia (LLV)")
st.write("Enter the patient's clinical data:")

# Input variables
sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
age = st.number_input("Age", min_value=0, max_value=100, value=40)
mode = st.selectbox("Transmission mode", [1, 2, 3, 4], format_func=lambda x: ["Homo/Bisexual", "IDU", "Heterosexual", "Other/Unknown"][x-1])
origin = st.selectbox("Country of origin", [1, 2, 3], format_func=lambda x: ["Spain", "Not Spain", "Unknown"][x-1])
edu = st.selectbox("Education level", [1, 2, 3], format_func=lambda x: ["Primary", "University/Higher", "Unknown"][x-1])
aids = st.selectbox("Previous AIDS diagnosis", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
art_cat = st.selectbox("Year of ART initiation", [1, 2, 3, 4], format_func=lambda x: ["2004–2007", "2008–2011", "2012–2015", "≥2016"][x-1])
hbv = st.selectbox("Hepatitis B coinfection", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
hcv = st.selectbox("Hepatitis C coinfection", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
vl_cat = st.selectbox("Initial viral load", [1, 2, 3], format_func=lambda x: ["<100,000", "≥100,000", "Unknown"][x-1])
cd4 = st.number_input("CD4 count (baseline)", min_value=0, max_value=2000, value=500)
tipo_tar = st.selectbox("Final ART regimen", [1, 2, 3, 4], format_func=lambda x: ["2NRTI+1NNRTI", "2NRTI+1PI", "2NRTI+1INSTI", "Other"][x-1])

# Predict button
if st.button("Estimate risk"):
    input_data = np.array([[sex, age, mode, origin, edu, aids, art_cat, hbv, hcv, vl_cat, cd4, tipo_tar]])
    proba = model.predict_proba(input_data)[0, 1]

    st.metric("Predicted risk of virological failure", f"{proba:.2%}")

    if proba < 0.01:
        st.success("🟢 Low risk of virological failure")
    else:
        st.warning("🔴 At risk of virological failure – closer monitoring recommended")
