# 🏠 House Price Prediction using XGBoost

## 📌 Project Overview

This project predicts house prices using Machine Learning. An XGBoost Regressor model is trained on housing data and deployed using Streamlit to provide real-time predictions through a simple web interface.

## 🚀 Features

- Data preprocessing and cleaning
- Categorical feature encoding using One-Hot Encoding
- House price prediction using XGBoost Regressor
- Model evaluation using R² Score, MAE, RMSE, and MAPE
- Interactive Streamlit web application
- Download prediction results as CSV

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Joblib
- Streamlit

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| R² Score | 0.7676 |
| MAE | 12932.99 |
| RMSE | 25638.03 |
| MAPE | 7.67% |

## 📂 Project Structure

```
HousePricePrediction/
│
├── house.py
├── app.py
├── house_price_model.pkl
├── features.pkl
├── sample_input.csv
├── requirements.txt
├── README.md
└── HousePricePrediction.csv
```

## ⚙️ Installation

```bash
git clone <repository-url>
cd HousePricePrediction

pip install -r requirements.txt
```

## ▶️ Run Application

```bash
streamlit run app.py
```

## 📈 Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Encoding
4. Train XGBoost Model
5. Evaluate Model
6. Save Model (.pkl)
7. Build Streamlit App
8. Generate Predictions

## 👨‍💻 Author

Muhammad Aslam Shanavas
B.Tech Computer Science Engineering
AISAT