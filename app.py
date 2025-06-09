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
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 6, 3)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 5, 2)
    toilets = st.sidebar.slider("Number of Toilets", 1, 5, 2)
    parking_space = st.sidebar.selectbox("Parking Space", [0, 1])
    year_added = st.sidebar.slider("Year Added", 2019, 2025, 2023)

    location_encoded = {'Ikeja': 0, 'Lekki': 1, 'Yaba': 2, 'Surulere': 3, 'Ajah': 4, 'Other': 5}[location]

    data = {
        'LOCATION': location_encoded,
        'BEDROOMS': bedrooms,
        'BATHROOMS': bathrooms,
        'TOILETS': toilets,
        'PARKING SPACE': parking_space,
#        'YEAR ADDED': year_added
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Show summary
st.subheader("Property Summary")
st.write(input_df)

# Prediction
if st.button("Predict Rent Price"):
    st.write("Input features:", input_df.columns.tolist())
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Rent: ₦{int(prediction):,}")
