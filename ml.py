import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("ML Model Prediction App")

# Input fields (adjust based on your model features)

cem = st.number_input("Enter cement amount:")
blastf = st.number_input("Enter blast furnace slag amount:")
flyas = st.number_input("Enter fly ash amount:")
water = st.number_input("Enter water amount:")
superplaster = st.number_input("Enter superplasticizer amount:")
courseagg = st.number_input("Enter coarse aggregate amount:")
fineagg = st.number_input("Enter fine aggregate amount:")
age = st.number_input("Enter age:")


# Prediction
if st.button("Predict"):
    input_data = np.array([[cem,blastf,flyas,water,superplaster,courseagg,fineagg,age]])  # shape must match model input
    prediction = model.predict(input_data).reshape(-1, 1)  # Reshape if necessary
    st.success(f"Prediction: {prediction[0]}")
