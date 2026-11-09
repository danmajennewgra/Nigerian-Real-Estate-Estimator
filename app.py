import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="Nigerian Real Estate Price Estimator",
    page_icon="🏠",
    layout="centered"
)

locations = {
    "Lagos": {"Lekki": 4.5, "Victoria Island": 8.0, "Ikoyi": 12.0, "Ikeja": 3.5, "Yaba": 2.0, "Ajah": 2.2},
    "Abuja": {"Maitama": 10.0, "Asokoro": 9.0, "Wuse": 5.0, "Gwarinpa": 2.5, "Lugbe": 1.2},
    "Rivers": {"GRA Phase 2": 3.5, "Peter Odili": 2.8, "Trans Amadi": 2.0, "Ada George": 1.3},
    "Oyo": {"Bodija": 1.5, "Jericho": 1.8, "Akobo": 1.0, "Challenge": 0.8}
}

property_types = {
    "Detached Duplex": 2.5,
    "Semi-Detached Duplex": 1.8,
    "Terraced Duplex": 1.5,
    "Block of Flats": 1.0,
    "Bungalow": 1.1
}

@st.cache_resource
def load_or_train_model():
    # Load existing model if present and compatible
    if os.path.exists("nigerian_real_estate_model.pkl"):
        try:
            return joblib.load("nigerian_real_estate_model.pkl")
        except Exception:
            pass

    # Otherwise generate cleaned dataset and train model automatically
    np.random.seed(42)
    rows = []
    for _ in range(1500):
        state = np.random.choice(list(locations.keys()), p=[0.45, 0.25, 0.15, 0.15])
        town = np.random.choice(list(locations[state].keys()))
        prop_type = np.random.choice(list(property_types.keys()))
        listing_type = np.random.choice(["Rent", "Sale"], p=[0.7, 0.3])
        bedrooms = np.random.randint(1, 7)
        bathrooms = bedrooms + np.random.randint(0, 2)
        toilets = bathrooms + 1
        parking_space = np.random.randint(1, 6)

        base_price = 600_000 * locations[state][town] * property_types[prop_type] * (1 + 0.25 * bedrooms)
        if listing_type == "Sale":
            base_price *= np.random.uniform(15, 22)

        noise = np.random.normal(1.0, 0.15)
        price_ngn = round(base_price * max(noise, 0.5), -4)
        rows.append([state, town, prop_type, listing_type, bedrooms, bathrooms, toilets, parking_space, price_ngn])

    df = pd.DataFrame(rows, columns=[
        "state", "town", "property_type", "listing_type",
        "bedrooms", "bathrooms", "toilets", "parking_space", "price_ngn"
    ])
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"] + df["toilets"]
    df["bath_bed_ratio"] = round(df["bathrooms"] / df["bedrooms"], 2)

    X = df.drop(columns=["price_ngn"])
    y = df["price_ngn"]

    categorical_cols = ["state", "town", "property_type", "listing_type"]
    numerical_cols = ["bedrooms", "bathrooms", "toilets", "parking_space", "total_rooms", "bath_bed_ratio"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, random_state=42))
    ])

    full_model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    full_model.fit(X, y)
    joblib.dump(full_model, "nigerian_real_estate_model.pkl")
    return full_model

model = load_or_train_model()

st.title("🏠 Nigerian Real Estate & Rent Estimator")
st.write("Estimate annual rent or property sale prices in Naira (₦) across major Nigerian cities.")

col1, col2 = st.columns(2)

with col1:
    listing_type = st.selectbox("Listing Type", ["Rent", "Sale"])
    state = st.selectbox("State", list(locations.keys()))
    town = st.selectbox("Town / Neighborhood", list(locations[state].keys()))
    property_type = st.selectbox("Property Type", list(property_types.keys()))

with col2:
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=10, value=3)
    toilets = st.slider("Toilets", min_value=1, max_value=12, value=4)
    parking_space = st.slider("Parking Spaces", min_value=0, max_value=10, value=2)

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
    prediction = model.predict(input_df)[0]

    if listing_type == "Rent":
        st.success(f"Estimated Annual Rent: ₦{prediction:,.0f} / year")
        st.info(f"Monthly Breakdown: ₦{prediction / 12:,.0f} / month")
    else:
        st.success(f"Estimated Sale Price: ₦{prediction:,.0f}")
        st.info(f"Estimated Price per Bedroom: ₦{prediction / bedrooms:,.0f}")
