import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Lagos Rent Predictor", layout="centered")

# Load model
model = joblib.load("best_lgbm_model.pkl")

# Title
st.title("🏠 Lagos House Rent Prediction App")
st.markdown("Use this tool to predict house rent prices in Lagos based on property features.")

# Sidebar inputs
st.sidebar.header("Enter Property Details")

def user_input_features():
    location = st.sidebar.selectbox("Location", ['Ikeja', 'Lekki', 'Yaba', 'Surulere', 'Ajah', 'Other'])
    lga = st.sidebar.selectbox("LGA", ['Ikeja', 'Eti-Osa', 'Yaba', 'Surulere', 'Ajah', 'Other'])
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 6, 3)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
    toilets = st.sidebar.slider("Number of Toilets", 1, 5, 2)
    house_type = st.sidebar.selectbox("House Type", ['Flat', 'Self-Contain', 'Duplex', 'Bungalow', 'Mini Flat', 'Other'])
    year_added = st.sidebar.slider("Year Added", 2019, 2025, 2023)
    month_added = st.sidebar.slider("Month Added", 1, 12, 6)
    day_added = st.sidebar.slider("Day Added", 1, 31, 15)

    # Encode categorical fields if needed
    location_encoded = {'Ikeja': 0, 'Lekki': 1, 'Yaba': 2, 'Surulere': 3, 'Ajah': 4, 'Other': 5}[location]
    house_type_encoded = {'Flat': 0, 'Self-Contain': 1, 'Duplex': 2, 'Bungalow': 3, 'Mini Flat': 4, 'Other': 5}[house_type]
    lga_encoded = {'Ikeja': 0, 'Eti-Osa': 1, 'Yaba': 2, 'Surulere': 3, 'Ajah': 4, 'Other': 5}[lga]

    data = {
        'LOCATION': location_encoded,
        'BEDROOMS': bedrooms,
        'BATHROOMS': bathrooms,
        'TOILETS': toilets,
        'HOUSE_TYPE': house_type_encoded,
        'LGA': lga_encoded,
        'YEAR ADDED': year_added,
        'MONTH ADDED': month_added,
        'DAY ADDED': day_added
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Show summary
st.subheader("Property Summary")
st.write(input_df)

# Prediction
if st.button("Predict Rent Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Rent: ₦{int(prediction):,}")
