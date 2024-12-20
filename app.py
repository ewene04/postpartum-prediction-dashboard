import streamlit as st
import lightgbm as lgb
import tensorflow as tf
import pandas as pd

# Add Page Configuration and Styling
st.set_page_config(
    page_title="Postpartum Depression Prediction",
    page_icon="🤱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom styling with the new color scheme
st.markdown(
    """
    <style>
    /* Main Dashboard Title */
    .main-title {
        text-align: center;
        font-size: 45px;
        color: #d9046a; /* First color from the scheme */
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Streamlit sidebar styling */
    .css-1d391kg {
        background-color: #d8ecca !important; /* Second color from the scheme */
    }

    /* Button Styling */
    .stButton button {
        background-color: #fbd6bd !important; /* Third color */
        color: black !important;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }

    .stButton button:hover {
        background-color: #feaeab !important; /* Fourth color */
        color: white !important;
    }

    /* Success and Error Messages */
    .st-success {
        background-color: #c5eddd !important; /* First color */
        color: black;
    }

    .st-error {
        background-color: #feb9be !important; /* Fifth color */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Dashboard Title
st.markdown("<div class='main-title'>Postpartum Depression Prediction</div>", unsafe_allow_html=True)

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

maternal_ages = {
    "18-44": 0, "17 or younger": 17, "45 or older": 45
}

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

# Real-Time Prediction using LightGBM
if mode == "Real-Time (LightGBM)":
    st.title("Real-Time Prediction (LightGBM)")
    
    # Input sliders for user data
    depression_frequency = st.selectbox("How frequently do you feel depressed since birth?", list(depression_frequencies.keys()))
    household_income = st.selectbox("What is the total income you have in the past 12 months?", list(incomes.keys()))
    maternal_race = st.selectbox("What is the race of the mother?", list(maternal_races.keys()))
    maternal_age = st.selectbox("What is the age of the mother?", list(maternal_ages.keys()))
    paternal_education = st.selectbox("What is the educational level of the father?", list(educational_levels.keys()))
    maternal_education = st.selectbox("What is the educational levle of the mother?", list(educational_levels.keys()))
    state = st.selectbox("What is the current state you are staying in?", list(states.keys()))
    depression_after = st.selectbox("Do you feel depressed after giving birth?", list(depression_after_birth.keys()))
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "MH_PPDPR": [depression_frequencies[depression_frequency]],
            "INCOME8": [incomes[household_income]],
            "MAT_RACE_PU": [maternal_races[maternal_race]],
            "MAT_AGE_PU": [maternal_ages[maternal_age]],
            "PAT_ED": [educational_levels[paternal_education]],
            "MAT_ED": [educational_levels[maternal_education]],
            "STATE": [states[state]],
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
            required_columns = ["MH_PPDPR", "INCOME8", "MAT_RACE_PU", "MAT_AGE_PU", "PAT_ED", "MAT_ED", "STATE", "MH_PPDX"]
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Map input data to numerical values
                batch_data_numeric = batch_data.replace({
                    "STATE": states, 
                    "MAT_RACE_PU": maternal_races,
                    "MAT_ED": educational_levels,
                    "PAT_ED": educational_levels,
                    "INCOME8": incomes,
                    "MH_PPDPR": depression_frequencies,
                    "MH_PPDX": depression_after_birth,
                    "MAT_AGE_PU": maternal_ages
                })

                # Reorder columns to match the required order
                batch_data_numeric = batch_data_numeric[required_columns]

                # Predict using ANN and extract probability for the positive class
                raw_predictions = ann_model.predict(batch_data_numeric)
                if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
                    predictions = raw_predictions[:, 1]
                else:
                    predictions = raw_predictions.flatten()

                # Ensure predictions length matches the input
                if len(predictions) != len(batch_data_numeric):
                    st.error("Error: The number of predictions does not match the input data. Please verify your model.")
                else:
                    # Add predictions to the original dataframe
                    batch_data["Prediction"] = [
                        "High Risk of Postpartum Depression" if pred > 0.5 else "Low Risk of Postpartum Depression" 
                        for pred in predictions
                    ]
                    
                    # Display results
                    st.write("Prediction Results:")
                    st.dataframe(batch_data)

                    # Allow download of results
                    st.download_button(
                        label="Download Predictions",
                        data=batch_data.to_csv(index=False),
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
