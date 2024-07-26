import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import load_data, preprocess_data, load_scaler_params, scale_data
from utils.visualization import plot_feature_correlations, plot_pairwise_relationships

# Set page config
st.set_page_config(
    page_title="Metal Ion Concentration Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Metal Ion Concentration Prediction")
st.markdown("""
    Welcome to the Metal Ion Concentration Prediction web app! This app allows you to predict the concentration of metal ions in water based on electrochemical data. 
    You can upload your data and make predictions using a pre-trained model.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Make Predictions"])

if app_mode == "Make Predictions":
    st.header("📂 Upload Prediction Data")
    predict_file = st.file_uploader("Choose a CSV file for prediction", type="csv", key="predict")

    if predict_file is not None:
        predict_df = pd.read_csv(predict_file)
        st.write("Prediction Data Preview:")
        st.write(predict_df.head())

        if st.button("Predict"):
            with st.spinner("Making predictions..."):
                data = load_data(predict_file)
                processed_data = preprocess_data(data)
                X = processed_data.drop(columns=['ccd', 'cpb', 'ccu'])

                # Load the pre-trained model and scaler parameters
                model = load_model('models/best_model_6.h5')
                scaler_min, scaler_max = load_scaler_params('models/scaler_params.npz')
                X_scaled = scale_data(X, scaler_min, scaler_max)

                predictions = model.predict(X_scaled)

            st.success("Prediction complete!")

            # Display predictions
            st.header("📈 Prediction Results")
            prediction_df = pd.DataFrame(predictions, columns=['Predicted Cd(II) Concentration', 'Predicted Pb(II) Concentration', 'Predicted Cu(II) Concentration'])
            st.write(prediction_df)

            # Download prediction results
            csv = prediction_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Prediction Results", csv, "prediction_results.csv", "text/csv")
