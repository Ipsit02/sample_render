import streamlit as st
import numpy as np
import xgboost as xgb
import pickle
# Load the trained model
model_path = 'xgb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Electricity Price Prediction using XGBoost")

# Input fields for the 9 features
DayOfWeek = st.number_input("Day of Week (0 = Monday, 6 = Sunday)", min_value=0, max_value=6)
WeekOfYear = st.number_input("Week of Year", min_value=1, max_value=52)
Day = st.number_input("Day of Month", min_value=1, max_value=31)
Year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
PeriodOfDay = st.number_input("Period of Day (e.g. 0-23 for hours)", min_value=0, max_value=23)

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
