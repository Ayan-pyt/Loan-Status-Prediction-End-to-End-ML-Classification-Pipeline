import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load model components with error handling
try:
    with open(os.path.join(MODEL_DIR, 'gb_model.pkl'), 'rb') as f:
        model = cloudpickle.load(f)
    with open(os.path.join(MODEL_DIR, 'gb_scaler.pkl'), 'rb') as f:
        scaler = cloudpickle.load(f)
    with open(os.path.join(MODEL_DIR, 'gb_encoder.pkl'), 'rb') as f:
        encoder = cloudpickle.load(f)
    with open(os.path.join(MODEL_DIR, 'column_info.pkl'), 'rb') as f:
        col_info = cloudpickle.load(f)
    
    st.success("✅ Model loaded successfully")
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("Please check that the model has been trained and saved correctly.")
    st.stop()

st.set_page_config(page_title="Loan Status Prediction", layout="wide")
st.title("🏦 Loan Status Prediction")
st.markdown("---")
st.subheader("📋 Loan Details")

# Create columns for numeric features
col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        help="Amount of loan requested"
    )
    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=5.0,
        step=0.1,
        help="Loan interest rate"
    )
    installment = st.number_input(
        "Monthly Installment ($)",
        min_value=0.0,
        value=350.0,
        step=10.0,
        help="Monthly payment amount"
    )

with col2:
    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Enter your annual income"
    )
    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
        help="Monthly debt divided by monthly income"
    )
    tot_cur_bal = st.number_input(
        "Total Current Balance ($)",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        help="Total current outstanding balance"
    )

with col3:
    fico_range_low = st.number_input(
        "FICO Score",
        min_value=300,
        max_value=850,
        value=700,
        help="Your credit score"
    )
    term = st.selectbox(
        "Loan Term",
        options=["36 months", "60 months"],
        help="Loan repayment period"
    )
    emp_length = st.selectbox(
        "Employment Length",
        options=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        index=5,
        help="Years of employment"
    )

st.markdown("---")
st.subheader("🏠 Loan Purpose & Verification")

col4, col5, col6 = st.columns(3)

with col4:
    home_ownership = st.selectbox(
        "Home Ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
        help="Current housing status"
    )

with col5:
    verification_status = st.selectbox(
        "Verification Status",
        options=["Verified", "Source Verified", "Not Verified"],
        help="Income verification status"
    )

with col6:
    purpose = st.selectbox(
        "Loan Purpose",
        options=["debt_consolidation", "credit_card", "home_improvement", "other", "business", "personal", "auto"],
        help="Primary purpose of the loan"
    )

st.markdown("---")
st.subheader("📝 Additional Information")

col7, col8 = st.columns(2)

with col7:
    title = st.text_input(
        "Loan Title",
        value="Personal Loan",
        help="Brief description of the loan"
    )

with col8:
    zip_code = st.text_input(
        "Zip Code",
        value="10000",
        help="5-digit zip code or first digit"
    )

st.markdown("---")

if st.button("🔮 Predict Loan Status", use_container_width=True):
    try:
        # Create input dataframe with numeric features in the exact order: 
        # ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'fico_range_low', 'tot_cur_bal']
        numeric_features = {
            'loan_amnt': [loan_amnt],
            'int_rate': [int_rate],
            'installment': [installment],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'fico_range_low': [fico_range_low],
            'tot_cur_bal': [tot_cur_bal]
        }
        input_numeric = pd.DataFrame(numeric_features)
        
        # Verify we have all numeric features
        expected_numeric = col_info['numeric_cols']
        if not all(col in input_numeric.columns for col in expected_numeric):
            missing = [col for col in expected_numeric if col not in input_numeric.columns]
            st.error(f"Missing numeric features: {missing}")
            st.stop()
        
        # Reorder to match training order
        input_numeric = input_numeric[expected_numeric]
        
        # Scale numeric features
        input_numeric_scaled = scaler.transform(input_numeric)
        
        # Create a sparse matrix from scaled numeric features
        from scipy.sparse import csr_matrix, hstack
        X_numeric_sparse = csr_matrix(input_numeric_scaled)
        
        # Handle categorical features in correct order:
        # ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'title', 'zip_code']
        cat_cols = col_info['categorical_cols']
        
        input_cat = pd.DataFrame({
            'term': [term],
            'emp_length': [emp_length],
            'home_ownership': [home_ownership],
            'verification_status': [verification_status],
            'purpose': [purpose],
            'title': [title],
            'zip_code': [zip_code]
        })
        
        # Reorder to match training order
        input_cat = input_cat[cat_cols]
        
        # Transform categorical features
        X_cat_transformed = encoder.transform(input_cat)
        
        # Combine numeric and categorical features
        X_processed = hstack([X_numeric_sparse, X_cat_transformed])
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]

        # Display results
        st.success("✅ Prediction Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Loan Status", str(prediction).upper(), delta=None)
        with col2:
            confidence = max(prediction_proba) * 100
            st.metric("Confidence", f"{confidence:.1f}%", delta=None)

        # Show probability distribution
        st.subheader("Prediction Probabilities:")
        classes = model.classes_
        for cls, prob in zip(classes, prediction_proba):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please ensure all inputs are valid and try again.") 