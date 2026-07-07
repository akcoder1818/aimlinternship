import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("wine_model.keras")
scaler = joblib.load("scaler.pkl")

# ---------------- TITLE ----------------
st.title("🍷 Wine Quality Prediction using TensorFlow")
st.write("Enter the wine properties below and click **Predict**.")

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
    citric_acid = st.number_input("Citric Acid", value=0.00)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    pH = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Wine Quality"):

    sample = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    quality = np.argmax(prediction) + 3

    confidence = np.max(prediction) * 100

    st.success(f"🍷 Predicted Wine Quality : **{quality}**")
    st.info(f"Confidence : **{confidence:.2f}%**")

    st.subheader("Prediction Probability")

    for i, prob in enumerate(prediction[0]):
        st.write(f"Quality {i+3}")
        st.progress(float(prob))