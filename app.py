import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load Model and Features
model = joblib.load("house_price_model.pkl")
features = joblib.load("features.pkl")

# Title
st.title("🏠 House Price Prediction")
st.markdown(
    """
    Upload a CSV file containing house data and get instant price predictions
    using a trained XGBoost model.
    """
)

# Sidebar
st.sidebar.header("📊 Model Performance")

st.sidebar.metric("R² Score", "0.7676")
st.sidebar.metric("MAE", "12932.99")
st.sidebar.metric("RMSE", "25638.03")
st.sidebar.metric("MAPE", "7.67%")

st.sidebar.markdown("---")
st.sidebar.write(
    "Model: XGBoost Regressor"
)

# File Upload
uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.dataframe(data.head())

        # Select only required columns
        data = data[features]

        # Prediction
        predictions = model.predict(data)

        # Create Result DataFrame
        result = pd.DataFrame({
            "Predicted Price": predictions
        })

        st.subheader("💰 Predicted House Prices")
        st.dataframe(result)

        # Download Button
        csv = result.to_csv(index=False)

        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Statistics
        st.subheader("📈 Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Minimum Price",
                f"${result['Predicted Price'].min():,.2f}"
            )

        with col2:
            st.metric(
                "Average Price",
                f"${result['Predicted Price'].mean():,.2f}"
            )

        with col3:
            st.metric(
                "Maximum Price",
                f"${result['Predicted Price'].max():,.2f}"
            )

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")

st.markdown("""
### About This Project

This application predicts house prices using a trained
**XGBoost Regressor** model.

#### Features Used
- MSSubClass
- LotArea
- OverallCond
- YearBuilt
- YearRemodAdd
- TotalBsmtSF
- Building Type
- Lot Configuration
- Exterior Type
- Zoning Information

#### Workflow
1. Data Preprocessing
2. Feature Engineering
3. XGBoost Model Training
4. Model Evaluation
5. Streamlit Deployment

**Developed using Python, Scikit-Learn, XGBoost, Pandas, and Streamlit.**
""")