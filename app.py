import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Nigerian Real Estate Price Estimator",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("nigerian_real_estate_model.pkl")

try:
    model = load_model()
except Exception as e:
    model = None

locations = {
    "Lagos": ["Lekki", "Victoria Island", "Ikoyi", "Ikeja", "Yaba", "Ajah"],
    "Abuja": ["Maitama", "Asokoro", "Wuse", "Gwarinpa", "Lugbe"],
    "Rivers": ["GRA Phase 2", "Peter Odili", "Trans Amadi", "Ada George"],
    "Oyo": ["Bodija", "Jericho", "Akobo", "Challenge"]
}

property_types = [
    "Detached Duplex",
    "Semi-Detached Duplex",
    "Terraced Duplex",
    "Block of Flats",
    "Bungalow"
]

st.title("🏠 Nigerian Real Estate & Rent Estimator")
st.write("Estimate annual rent or property sale prices in Naira (₦) across major Nigerian cities.")

col1, col2 = st.columns(2)

df_clean = pd.read_csv("nigerian_real_estate_cleaned.csv") if pd.io.common.file_exists("nigerian_real_estate_cleaned.csv") else None

with col1:
    listing_type = st.selectbox("Listing Type", ["Rent", "Sale"])
    state = st.selectbox("State", list(locations.keys()))
    town = st.selectbox("Town / Neighborhood", locations[state])
    property_type = st.selectbox("Property Type", property_types)

with col2:
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=10, value=3)
    toilets = st.slider("Toilets", min_value=1, max_value=12, value=4)
    parking_space = st.slider("Parking Spaces", min_value=0, max_value=10, value=2)

# Engineer the exact features used during model training
total_rooms = bedrooms + bathrooms + toilets
bath_bed_ratio = round(bathrooms / bedrooms, 2)

input_df = pd.DataFrame([{
    "state": state,
    "town": town,
    "property_type": property_type,
    "listing_type": listing_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "toilets": toilets,
    "parking_space": parking_space,
    "total_rooms": total_rooms,
    "bath_bed_ratio": bath_bed_ratio
}])

if st.button("Estimate Price in Naira (₦)", use_container_width=True):
    if model is not None:
        prediction = model.predict(input_df)[0]
        if listing_type == "Rent":
            st.success(f"Estimated Annual Rent: ₦{prediction:,.0f} / year")
            st.info(f"Monthly Breakdown: ₦{prediction / 12:,.0f} / month")
        else:
            st.success(f"Estimated Sale Price: ₦{prediction:,.0f}")
    else:
        st.warning("Model file 'nigerian_real_estate_model.pkl' not found. Please train and save the model first.")
