import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# Load models
@st.cache_resource
def load_models():
    lgb_model = pickle.load(open("lightgbm_model.pkl", "rb"))
    ann_model = tf.keras.models.load_model("ann_model.h5")
    return lgb_model, ann_model

lgb_model, ann_model = load_models()

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

if mode == "Real-Time (LightGBM)":
    st.title("Real-Time Prediction (LightGBM)")
    age = st.slider("Age", 15, 50, 30)
    num_children = st.slider("Number of Children", 0, 10, 1)
    stress_level = st.slider("Stress Level (0-100)", 0, 100, 50)
    data = pd.DataFrame({"Age": [age], "Number of Children": [num_children], "Stress Level": [stress_level]})

    if st.button("Predict"):
        prediction = lgb_model.predict(data)[0]
        if prediction > 0.5:
            st.error("High Risk of Postpartum Depression")
        else:
            st.success("Low Risk of Postpartum Depression")

elif mode == "Batch (ANN)":
    st.title("Batch Prediction (ANN)")
    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        predictions = ann_model.predict(batch_data)
        batch_data["Prediction"] = predictions
        st.write("Predictions:")
        st.write(batch_data)
        st.download_button("Download Results", batch_data.to_csv(index=False), "predictions.csv", "text/csv")
