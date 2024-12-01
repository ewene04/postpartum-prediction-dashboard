import streamlit as st
import lightgbm as lgb
import tensorflow as tf
import pandas as pd

# Cache resource to load models
@st.cache_resource
def load_models():
    try:
        # Load the LightGBM model from .txt
        lgb_model = lgb.Booster(model_file="lightgbm_model.txt")
        
        # Load the ANN model
        ann_model = tf.keras.models.load_model("ann_model.h5")
        
        return lgb_model, ann_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
lgb_model, ann_model = load_models()

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode:", ["Real-Time (LightGBM)", "Batch (ANN)"])

# Real-Time Prediction using LightGBM
if mode == "Real-Time (LightGBM)":
    st.title("Real-Time Prediction (LightGBM)")
    
    # Input sliders for user data
    age = st.slider("Age", 15, 50, 30)
    num_children = st.slider("Number of Children", 0, 10, 1)
    stress_level = st.slider("Stress Level (0-100)", 0, 100, 50)
    
    # Input DataFrame
    data = pd.DataFrame({
        "Age": [age], 
        "Number of Children": [num_children], 
        "Stress Level": [stress_level]
    })

    if st.button("Predict"):
        if lgb_model is not None:
            try:
                # Predict using LightGBM
                prediction = lgb_model.predict(data)[0]
                if prediction > 0.5:
                    st.error("High Risk of Postpartum Depression")
                else:
                    st.success("Low Risk of Postpartum Depression")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Model not loaded. Please check your model files.")

# Batch Prediction using ANN
elif mode == "Batch (ANN)":
    st.title("Batch Prediction (ANN)")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV File", type="csv")
    
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.write(batch_data.head())
            
            if ann_model is not None:
                # Predict using ANN
                predictions = ann_model.predict(batch_data)
                batch_data["Prediction"] = predictions
                st.write("Predictions:")
                st.write(batch_data)
                
                # Download button for results
                st.download_button(
                    label="Download Results",
                    data=batch_data.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Model not loaded. Please check your model files.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
