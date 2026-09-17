import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("iris_model.pkl")

# Page title
st.title("Iris Flower Species Prediction")

# Input widgets for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Prediction button
if st.button("Predict Species"):
    # Create a numpy array from the inputs
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]]).astype(np.float64)

    # Make prediction
    prediction = model.predict(features)

    # Display the prediction
    st.success(f"The predicted Iris species is: {prediction[0]}")
