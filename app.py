import streamlit as st
import lightgbm as lgb
import tensorflow as tf
import pandas as pd

@st.cache_resource
def load_models():
    # Load LightGBM model
    lgb_model = lgb.Booster(model_file="lightgbm_model.txt")
    # Load ANN model
    ann_model = tf.keras.models.load_model("ann_model.h5")
    return lgb_model, ann_model

# Load the models
lgb_model, ann_model = load_models()

# Dropdown options and mappings
states = {
    "Not sure": 0, "Utah": 1, "New York City": 2, "Illinois": 3, "Colorado": 4, 
    "Nebraska": 5, "New Jersey": 6, "Missouri": 7, "Pennsylvania": 8, 
    "Maryland": 9, "Puerto Rico": 10, "New York (State)": 11, "Alabama": 12, 
    "New Hampshire": 13, "Washington, D.C.": 14, "Arizona": 15
}

maternal_races = {
    "Not sure": 0, "WHITE": 2, "BLACK": 3, "AM INDIAN": 4, "AK NATIVE": 10, 
    "ASIAN": 21, "HAWAIIAN/OTH PAC ISLANDER": 22, "OTHER/MULTIPLE RACE": 23
}

educational_levels = {
    "Not sure": 0, "<= 8TH GRADE": 1, "9-12 GRADE, NO DIPLOMA": 2, 
    "HIGH SCHOOL GRAD/GED": 3, "SOME COLLEGE, NO DEG/ASSOCIATE DEG": 4, 
    "BACHELORS/MASTERS/DOCTORATE/PROF": 5
}

incomes = {
    "Not sure": 0, "$0 TO $16,000": 1, "$16,001 TO $20,000": 2, "$20,001 TO $24,000": 3, 
    "$24,001 TO $28,000": 4, "$28,001 TO $32,000": 5, "$32,001 TO $40,000": 6, 
    "$40,001 TO $48,000": 7, "$48,001 TO $57,000": 8, "$57,001 TO $60,000": 9, 
    "$60,001 TO $73,000": 10, "$73,001 TO $85,000": 11, "$85,001 TO $99,999": 12, 
    "$100,001 OR MORE": 13
}

depression_frequencies = {
    "Not sure": 0, "ALWAYS": 1, "OFTEN": 2, "SOMETIMES": 3, "RARELY": 4, "NEVER": 5
}

depression_after_birth = {
    "Not sure": 0, "NO": 1, "YES": 2
}

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

# Real-Time Prediction using LightGBM
if mode == "Real-Time (LightGBM)":
    st.title("Real-Time Prediction (LightGBM)")
    
    # Input sliders for user data
    state = st.selectbox("What is the current state you are staying in?", list(states.keys()))
    maternal_race = st.selectbox("What is your age?", list(maternal_races.keys()))
    maternal_education = st.selectbox("What is your wife's educational level?", list(educational_levels.keys()))
    paternal_education = st.selectbox("What is your husband's educational level?", list(educational_levels.keys()))
    household_income = st.selectbox("What is the total income you have in the past 12 months?", list(incomes.keys()))
    depression_frequency = st.selectbox("How frequently do you feel depressed since birth?", list(depression_frequencies.keys()))
    depression_after = st.selectbox("Do you feel depressed after giving birth?", list(depression_after_birth.keys()))
    
    if st.button("Predict"):
        # Map inputs to numerical values
        input_data = pd.DataFrame({
            "STATE": [states[state]],
            "MAT_RACE_PU": [maternal_races[maternal_race]],
            "MAT_ED": [educational_levels[maternal_education]],
            "PAT_ED": [educational_levels[paternal_education]],
            "INCOME8": [incomes[household_income]],
            "MH_PPDPR": [depression_frequencies[depression_frequency]],
            "MH_PPDX": [depression_after_birth[depression_after]]
        })

        try:
            # Predict using LightGBM
            prediction = lgb_model.predict(input_data)[0]
            if prediction > 0.5:
                st.error("High Risk of Postpartum Depression")
            else:
                st.success("Low Risk of Postpartum Depression")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Batch Prediction using ANN
elif mode == "Batch (ANN)":
    st.title("Batch Prediction (ANN)")
    
    uploaded_file = st.file_uploader("Upload a CSV File (8 columns required)", type="csv")
    
    if uploaded_file:
        try:
            # Load uploaded data
            batch_data = pd.read_csv(uploaded_file)

            # Validate column structure
            required_columns = ["STATE", "MAT_RACE_PU", "MAT_ED", "PAT_ED", "INCOME8", "MH_PPDPR", "MH_PPDX"]
            if not set(required_columns).issubset(batch_data.columns):
                st.error("Uploaded file must contain the following columns: " + ", ".join(required_columns))
            else:
                # Predict using ANN
                batch_data_numeric = batch_data.replace({
                    "STATE": states, 
                    "MAT_RACE_PU": maternal_races,
                    "MAT_ED": educational_levels,
                    "PAT_ED": educational_levels,
                    "INCOME8": incomes,
                    "MH_PPDPR": depression_frequencies,
                    "MH_PPDX": depression_after_birth
                })

                predictions = ann_model.predict(batch_data_numeric)
                batch_data["Prediction"] = [
                    "High Risk of Postpartum Depression" if pred > 0.5 else "Low Risk of Postpartum Depression" 
                    for pred in predictions
                ]
                st.write("Prediction Results:")
                st.dataframe(batch_data)

                # Download the results
                st.download_button(
                    label="Download Predictions",
                    data=batch_data.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
