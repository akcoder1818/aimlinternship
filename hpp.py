import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="House Rent Prediction")

st.title("🏠 House Rent Prediction")

# Load saved objects
hpp = joblib.load("house_rent_prediction.pkl")

encoder = hpp["encoder"]
scaler = hpp["scaler"]
model = hpp["model"]

# Inputs
BHK = st.number_input("Enter BHK", min_value=1, max_value=10, value=2)
Size = st.number_input("Enter Size", min_value=100, max_value=10000, value=1200)

Area_Type = st.selectbox(
    "Area Type",
    ["Super Area", "Carpet Area", "Built Area"]
)

City = st.selectbox(
    "City",
    ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]
)

Furnishing_Status = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

Tenant_Preferred = st.selectbox(
    "Tenant Preferred",
    ["Family", "Bachelors", "Any"]
)

Bathroom = st.number_input(
    "Bathroom",
    min_value=1,
    max_value=10,
    value=2
)

Point_of_Contact = st.selectbox(
    "Point of Contact",
    ["Contact Owner", "Contact Agent", "Contact Builder"]
)

if st.button("Predict Rent"):

    input_data = pd.DataFrame({
        "BHK": [BHK],
        "Size": [Size],
        "Area Type": [Area_Type],
        "City": [City],
        "Furnishing Status": [Furnishing_Status],
        "Tenant Preferred": [Tenant_Preferred],
        "Bathroom": [Bathroom],
        "Point of Contact": [Point_of_Contact]
    })

    # Encode
    input_data = encoder.transform(input_data)

    # Scale
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    st.success(f"Predicted Rent: ₹ {prediction[0]:,.2f}")