import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression 

# Define the correct model filename
MODEL_FILENAME = 'model-reg-67130701708.pkl'

# --- MODEL LOADING FUNCTION ---
@st.cache_resource
def load_trained_model():
    """Loads the pre-trained model from the .pkl file."""
    try:
        # Step 1: Load the model using pickle
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file '{MODEL_FILENAME}' was not found. Please ensure it is uploaded to your GitHub repository.")
        
        # Fallback: Create a temporary model to allow the UI to display 
        temp_model = LinearRegression()
        temp_model.fit(pd.DataFrame({'youtube': [1], 'tiktok': [1], 'instagram': [1]}), pd.Series([10]))
        return temp_model
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        return None

# Load the model
model = load_trained_model()
model_loaded = (model is not None)

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Sales Prediction Regression App", layout="wide")

st.title("🎯 Sales Prediction Deployment")
st.markdown("### Linear Regression Model to Estimate Sales based on Advertising Budget")

if model_loaded:
    st.markdown("---")
    st.subheader("Advertising Budget Inputs (in Thousands)")
    
    # 1. Get user inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        youtube_budget = st.slider("YouTube Budget", min_value=0, max_value=300, value=50, step=1)
        
    with col2:
        tiktok_budget = st.slider("TikTok Budget", min_value=0, max_value=50, value=50, step=1)
        
    with col3:
        instagram_budget = st.slider("Instagram Budget", min_value=0, max_value=150, value=50, step=1)

    # 2. Prepare the input data for prediction
    input_data = pd.DataFrame({
        'youtube': [youtube_budget],
        'tiktok': [tiktok_budget],
        'instagram': [instagram_budget]
    })
    
    st.markdown("---")
    
    # 3. Create the prediction button
    if st.button("Predict Estimated Sales", help="Click to generate sales prediction based on inputs", type="primary"):
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        st.success(f"### Predicted Sales Value:")
        st.balloons()
        
        st.metric(
            label="Estimated Sales",
            value=f"${prediction[0]:,.2f}",
            delta="Predicted result from the trained Linear Regression Model."
        )
        
        st.markdown(
            """
            *หมายเหตุ: ตัวเลขที่แสดงเป็นค่าทำนายยอดขายโดยรวม โดย Model ถูกฝึกด้วยข้อมูลงบประมาณ **YouTube, TikTok, และ Instagram** เป็นปัจจัยหลัก*
            """
        )
