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

# Dropdown options and mappings for all fields
options_and_mappings = {
    "STATE": {
        "options": [
            "Not sure", "Utah", "New York City", "Illinois", "Colorado", "Nebraska",
            "New Jersey", "Missouri", "Pennsylvania", "Maryland", "Puerto Rico",
            "New York (State)", "Alabama", "New Hampshire", "Washington, D.C.", "Arizona"
        ],
        "mapping": {
            "Not sure": 0, "Utah": 1, "New York City": 2, "Illinois": 3, "Colorado": 4, "Nebraska": 5,
            "New Jersey": 6, "Missouri": 7, "Pennsylvania": 8, "Maryland": 9, "Puerto Rico": 10,
            "New York (State)": 11, "Alabama": 12, "New Hampshire": 13, "Washington, D.C.": 14, "Arizona": 15
        }
    },
    "MAT_RACE_PU": {
        "options": [
            "Not sure", "WHITE", "BLACK", "AM INDIAN", "AK NATIVE", "ASIAN",
            "HAWAIIAN/OTH PAC ISLANDER", "OTHER/MULTIPLE RACE"
        ],
        "mapping": {
            "Not sure": 0, "WHITE": 1, "BLACK": 2, "AM INDIAN": 3, "AK NATIVE": 10,
            "ASIAN": 21, "HAWAIIAN/OTH PAC ISLANDER": 22, "OTHER/MULTIPLE RACE": 23
        }
    },
    "MAT_ED": {
        "options": [
            "Not sure", "<= 8TH GRADE", "9-12 GRADE, NO DIPLOMA", "HIGH SCHOOL GRAD/GED",
            "SOME COLLEGE, NO DEG/ASSOCIATE DEG", "BACHELORS/MASTERS/DOCTORATE/PROF"
        ],
        "mapping": {
            "Not sure": 0, "<= 8TH GRADE": 1, "9-12 GRADE, NO DIPLOMA": 2,
            "HIGH SCHOOL GRAD/GED": 3, "SOME COLLEGE, NO DEG/ASSOCIATE DEG": 4,
            "BACHELORS/MASTERS/DOCTORATE/PROF": 5
        }
    },
    "PAT_ED": {
        "options": [
            "Not sure", "<= 8TH GRADE", "9-12 GRADE, NO DIPLOMA", "HIGH SCHOOL GRAD/GED",
            "SOME COLLEGE, NO DEG/ASSOCIATE DEG", "BACHELORS/MASTERS/DOCTORATE/PROF"
        ],
        "mapping": {
            "Not sure": 0, "<= 8TH GRADE": 1, "9-12 GRADE, NO DIPLOMA": 2,
            "HIGH SCHOOL GRAD/GED": 3, "SOME COLLEGE, NO DEG/ASSOCIATE DEG": 4,
            "BACHELORS/MASTERS/DOCTORATE/PROF": 5
        }
    },
    "INCOME8": {
        "options": [
            "Not sure", "$0 TO $16,000", "$16,001 TO $20,000", "$20,001 TO $24,000",
            "$24,001 TO $28,000", "$28,001 TO $32,000", "$32,001 TO $40,000",
            "$40,001 TO $48,000", "$48,001 TO $57,000", "$57,001 TO $60,000",
            "$60,001 TO $73,000", "$73,001 TO $85,000", "$85,001 TO $99,999",
            "$100,001 OR MORE"
        ],
        "mapping": {
            "Not sure": 0, "$0 TO $16,000": 1, "$16,001 TO $20,000": 2, "$20,001 TO $24,000": 3,
            "$24,001 TO $28,000": 4, "$28,001 TO $32,000": 5, "$32,001 TO $40,000": 6,
            "$40,001 TO $48,000": 7, "$48,001 TO $57,000": 8, "$57,001 TO $60,000": 9,
            "$60,001 TO $73,000": 10, "$73,001 TO $85,000": 11, "$85,001 TO $99,999": 12,
            "$100,001 OR MORE": 13
        }
    },
    "MH_PPDPR": {
        "options": ["Not sure", "ALWAYS", "OFTEN", "SOMETIMES", "RARELY", "NEVER"],
        "mapping": {
            "Not sure": 0, "ALWAYS": 1, "OFTEN": 2, "SOMETIMES": 3, "RARELY": 4, "NEVER": 5
        }
    },
    "MH_PPDX": {
        "options": ["Not sure", "NO", "YES"],
        "mapping": {
            "Not sure": 0, "NO": 1, "YES": 2
        }
    }
}

# Real-Time Prediction Section
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

if mode == "Batch (ANN)":
    st.title("Batch Prediction (ANN)")
    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.write(batch_data.head())
            
            # Map categorical columns to numeric codes
            for column, value in options_and_mappings.items():
                batch_data[column] = batch_data[column].map(value["mapping"])
            
            predictions = ann_model.predict(batch_data)
            batch_data["Prediction"] = ["High Risk of Postpartum Depression" if p > 0.5 else "Low Risk of Postpartum Depression" for p in predictions]
            
            st.write("Predictions Table:")
            st.dataframe(batch_data)
            
            # Allow downloading the full dataset with predictions
            st.download_button(
                label="Download Results",
                data=batch_data.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
