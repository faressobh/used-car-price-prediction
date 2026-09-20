import numpy as np
import pandas as pd
import joblib
import streamlit as st
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("car_price_model.json")

scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
category_options = joblib.load("category_options.pkl")
preprocessing_stats = joblib.load("preprocessing_stats.pkl")

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗"
)

st.title("🚗 Used Car Price Prediction")

st.image("car.jpg", use_container_width=True)

st.write(
    "Select the car specifications to get the predicted price in Egyptian Pounds."
)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        category_options["Brand"]
    )

    model_name = st.selectbox(
        "Model",
        category_options["Model"]
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        category_options["Fuel Type"]
    )

    transmission_type = st.selectbox(
        "Transmission Type",
        category_options["Transmission Type"]
    )

with col2:
    body_type = st.selectbox(
        "Body Type",
        category_options["Body Type"]
    )

    kilometers = st.number_input(
        "Kilometers",
        min_value=0,
        value=50000,
        step=1000
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1970,
        max_value=2026,
        value=2018,
        step=1
    )

    engine_capacity = st.number_input(
        "Engine Capacity (CC)",
        min_value=0,
        value=1600,
        step=100
    )

if st.button("Predict Price"):

    car_age = 2026 - year

    input_row = {
        "Brand": brand,
        "Model": model_name,
        "Kilometers": kilometers,
        "Fuel Type": fuel_type,
        "Transmission Type": transmission_type,
        "Engine Capacity (CC)": engine_capacity,
        "Body Type": body_type,
        "Car_Age": car_age,
    }

    input_df = pd.DataFrame([input_row])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction_log = model.predict(input_df)[0]

    prediction = np.expm1(prediction_log)

    prediction = min(
        prediction,
        preprocessing_stats["price_95th"]
    )

    st.success(
        f"Predicted Price: {prediction:,.0f} EGP"
    )

    st.caption(
        "Predictions above the 95th percentile are capped at the maximum value "
        "because the model was trained on prices limited to this percentile."
    )