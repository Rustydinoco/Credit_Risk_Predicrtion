import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Credit Risk Prediction")

@st.cache_resource
def load_model():
    model = joblib.load("model_credit2.pkl")
    scaler = joblib.load("scaler_credit2.pkl")
    return model,scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'credit_risk_model.pkl' and 'scaler.pkl' are in the working directory.")
    st.stop()

st.sidebar.header("Customer Data Input") 

Income =st.sidebar.number_input("Income (Dollar)", min_value=0, max_value=1000000, value=0)
Debt =st.sidebar.number_input("Debt (Dollar)", min_value=0, max_value=1000000, value=0)
CreditScore = st.sidebar.number_input("Credit Score", min_value=0, max_value=850, value=0)
Age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=0)

if st.button("Predict Credit Risk"):
    input_data = pd.DataFrame({
        'umur': [Age],                
        'pendapatan_tahunan': [Income], 
        'skor_kredit': [CreditScore],    
        'jumlah_pinjaman': [Debt]       
    })

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 0:
        st.success("**APPROVED**")
        st.metric(label="Confidence Score",value=f"{np.max(model.predict_proba(scaled_data))*100:.2f}%")
    else:
        st.error("**REJECTED**")
        st.metric(label="Confidence Score",value=f"{np.max(model.predict_proba(scaled_data))*100:.2f}%")

