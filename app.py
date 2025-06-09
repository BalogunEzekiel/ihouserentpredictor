import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Lagos Rent Predictor", layout="centered")

model = joblib.load("best_lgbm_model.pkl")

st.title("🏠 Lagos House Rent Prediction App")
st.markdown("Use this tool to predict house rent prices in Lagos based on property features.")

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
        'BEDROOM': bedrooms,
        'BATHROOM': bathrooms,
        'TOILET': toilets,
        'PARKING SPACE': parking_space,
        'YEAR ADDED': year_added
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Property Summary")
st.write(input_df)

if st.button("Predict Rent Price"):
    model = joblib.load('house_rent_model.pkl')

# Input DataFrame
input_df = pd.DataFrame({
    'LOCATION': [location],
    'BEDROOMS': [bedrooms],
    'BATHROOMS': [bathrooms],
    'TOILETS': [toilets],
    'HOUSE_TYPE': [house_type]
})

# Predict using the entire pipeline
    prediction = model.predict(input_df)[0]
#    prediction = model.predict(input_df)
    st.success(f"Estimated Rent: ₦{int(prediction[0]):,}")
