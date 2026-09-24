import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("uber_fare_model_compressed.pkl")

# Page title
st.title("🚕 Uber Fare Prediction")

st.write("Enter the trip details below to predict the estimated fare.")

# User inputs
trip_distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.0,
    value=5.0
)

time_of_day = st.number_input(
    "Time of Day (Hour)",
    min_value=0,
    max_value=23,
    value=12
)

passenger_rating = st.number_input(
    "Passenger Rating",
    min_value=0.0,
    max_value=5.0,
    value=4.5
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Rain", "Snow"]
)

service_type = st.selectbox(
    "Service Type",
    ["Pool", "UberX", "Black"]
)

traffic_level = st.number_input(
    "Traffic Level",
    min_value=0.0,
    value=5.0
)

# Prediction button
if st.button("Predict Fare"):

    input_data = pd.DataFrame({
        "Trip_Distance_km": [trip_distance],
        "Time_of_Day": [time_of_day],
        "Passenger_Rating": [passenger_rating],
        "Weather": [weather],
        "Service_Type": [service_type],
        "Traffic_Level": [traffic_level]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Fare: {prediction:.2f}")
