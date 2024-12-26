import streamlit as st
import lightgbm as lgb
import tensorflow as tf
import pandas as pd
import json

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

# Load dropdown options from JSON file
with open("dropdown_options.json", "r") as file:
    dropdown_options = json.load(file)

# Assign to variables for usage in the app
states = dropdown_options["states"]
maternal_races = dropdown_options["maternal_races"]
educational_levels = dropdown_options["educational_levels"]
incomes = dropdown_options["incomes"]
depression_frequencies = dropdown_options["depression_frequencies"]
depression_after_birth = dropdown_options["depression_after_birth"]

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

# Real-Time Prediction using LightGBM
if mode == "Real-Time (LightGBM)":
    st.title("Real-Time Prediction (LightGBM)")
    
    # Input sliders for user data
    depression_frequency = st.selectbox("Since your new baby was born, how often have you felt down, depressed, or hopeless?", list(depression_frequencies.keys()))
    household_income = st.selectbox("During the 12 months before your new baby was born, what was your yearly total household income before taxes?", list(incomes.keys()))
    maternal_race = st.selectbox("What is your race/ethnicity?", list(maternal_races.keys()))
    maternal_age = st.number_input(
    "How old are you?",
    min_value=17,
    max_value=45,
    value=18,  # Default value
    step=1
)
    paternal_education = st.selectbox("What is the highest level of education completed by your baby's other parent or caregiver?", list(educational_levels.keys()))
    maternal_education = st.selectbox("What is the highest level of education you have completed?", list(educational_levels.keys()))
    state = st.selectbox("What is the current state you are living in?", list(states.keys()))
    depression_after = st.selectbox("Since your new baby was born, has a doctor, nurse, or other health care worker told you that you had depression?", list(depression_after_birth.keys()))
    
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "MH_PPDPR": [depression_frequencies[depression_frequency]],
            "INCOME8": [incomes[household_income]],
            "MAT_RACE_PU": [maternal_races[maternal_race]],
            "MAT_AGE_PU": [maternal_age],
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

import altair as alt

# Batch Prediction using ANN
if mode == "Batch (ANN)":
    st.title("Batch Prediction (ANN)")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload a CSV File (8 columns required)", type="csv")

    if uploaded_file:  # Ensure uploaded_file is only accessed when defined
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
                    "MH_PPDX": depression_after_birth
                })

                # Validate maternal age column
                if not pd.api.types.is_numeric_dtype(batch_data["MAT_AGE_PU"]):
                    st.error("Error: MAT_AGE_PU column must contain numeric values representing maternal age.")
                elif (batch_data["MAT_AGE_PU"] < 17).any() or (batch_data["MAT_AGE_PU"] > 45).any():
                    st.error("Error: MAT_AGE_PU values must be between 17 and 45.")
                else:
                    batch_data_numeric["MAT_AGE_PU"] = batch_data["MAT_AGE_PU"]

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
                        # Calculate metrics
                        total_cases = len(batch_data)
                        high_risk_cases = sum(batch_data["Prediction"] == "High Risk of Postpartum Depression")
                        low_risk_cases = total_cases - high_risk_cases

                        # Display metrics side by side with bold styling
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"<h5 style='text-align: center; font-weight: bold;'>Total Cases</h4>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='text-align: center; margin-top: -10px;'>{total_cases}</h2>", unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"<h5 style='text-align: center; font-weight: bold;'>Low Risk Cases</h4>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='text-align: center; margin-top: -10px;'>{low_risk_cases}</h2>", unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"<h5 style='text-align: center; font-weight: bold;'>High Risk Cases</h4>", unsafe_allow_html=True)
                            st.markdown(f"<h1 style='text-align: center; margin-top: -10px;'>{high_risk_cases}</h2>", unsafe_allow_html=True)

                        # Generate and display the pie chart with custom colors
                        chart_data = pd.DataFrame({
                            "Risk Type": ["Low Risk", "High Risk"],
                            "Count": [low_risk_cases, high_risk_cases]
                        })

                        custom_colors = alt.Scale(
                            domain=["Low Risk", "High Risk"],
                            range=["#41B7C4", "#F0007B"]  # Blue for Low Risk, Pink for High Risk
                        )

                        # Calculate percentages
                        chart_data["Percentage"] = (chart_data["Count"] / chart_data["Count"].sum() * 100).round(1)

                        pie_chart = alt.Chart(chart_data).mark_arc().encode(
                            theta=alt.Theta(field="Count", type="quantitative"),
                            color=alt.Color(field="Risk Type", type="nominal", scale=custom_colors),
                            tooltip=[
                                alt.Tooltip("Risk Type", title="Risk Type"),
                                alt.Tooltip("Count", title="Count"),
                                alt.Tooltip("Percentage", title="Percentage (%)")
                            ]
                        ).properties(
                            title="Risk Distribution"
                        )

                        # Display the pie chart
                        st.altair_chart(pie_chart, use_container_width=True)
                        
                        # Show prediction results at the end
                        st.write("**Prediction Results**")
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
