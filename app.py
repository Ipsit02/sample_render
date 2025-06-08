import streamlit as st
import numpy as np
import xgboost as xgb
import pickle
import gdown
import os

# Define file name and proper Google Drive direct download link (with uc?id=)
model_file = "xbg_pkl_model.pkl"
model_url = "https://drive.google.com/uc?id=1heih5b8ufHjKdUMIzJS7Oex_3NyizbTW"

# Download the model only if it's not already downloaded
if not os.path.exists(model_file):
    gdown.download(model_url, model_file, quiet=False)

# Load the model
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Electricity Price Prediction using XGBoost")

# Input fields
DayOfWeek = st.number_input("Day of Week (0 = Monday, 6 = Sunday)", min_value=0, max_value=6)
WeekOfYear = st.number_input("Week of Year", min_value=1, max_value=52)
Day = st.number_input("Day of Month", min_value=1, max_value=31)
Year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
PeriodOfDay = st.number_input("Period of Day (0-23)", min_value=0, max_value=23)

Forecasted_Wind_Production = st.number_input("Forecasted Wind Production (MW)")
Forecasted_Load = st.number_input("Forecasted Load (MW)")
Actual_Temperature = st.number_input("Actual Temperature (°C)")
CO2Intensity = st.number_input("CO2 Intensity (gCO2/kWh)")

# Predict button
if st.button("Predict Price"):
    try:
        input_features = np.array([[DayOfWeek, WeekOfYear, Day, Year, PeriodOfDay,
                                    Forecasted_Wind_Production, Forecasted_Load,
                                    Actual_Temperature, CO2Intensity]])
        prediction = model.predict(input_features)
        st.success(f"Predicted Electricity Price: €{round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"Error: {e}")
