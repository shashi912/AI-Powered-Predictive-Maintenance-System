import streamlit as st
import pandas as pd
import pickle

st.title("Predictive Maintenance Dashboard")

# Load trained model
model = pickle.load(open('outputs/model.pkl','rb'))

# Upload sensor data
uploaded_file = st.file_uploader("Upload Sensor Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = model.predict(df)
    df['Predicted_Failure'] = predictions
    st.subheader("Predictions")
    st.dataframe(df)

    st.subheader("Failure Count")
    st.bar_chart(df['Predicted_Failure'].value_counts())
