# 🏠 House Price Prediction using Machine Learning

This project is an end-to-end Machine Learning regression system that predicts house prices based on property features.

## 🌐 Live Demo
🔗 **Streamlit App:**  
https://house-price-predictiongit-78dzcnrbuuenmydgxuyar2.streamlit.app

## 📌 Problem Statement
Predict the price of a house using features such as area, number of rooms, facilities, and furnishing status.

## 🧠 Machine Learning Approach

- Regression problem (predicting a numeric value)
- Final model: **Multiple Linear Regression**
- Random Forest was also evaluated, but Multiple Linear Regression achieved better performance on this dataset


## 📂 Dataset
Kaggle Housing Price Dataset

### Features
- Area, bedrooms, bathrooms, stories
- Parking, main road access
- Air conditioning, preferred area
- Furnishing status

### Target
- Price

## ⚙️ Workflow
1. Data loading & exploration
2. Data preprocessing (one-hot encoding)
3. Train-test split
4. Model training
5. Model evaluation (MAE, MSE, R²)
6. Model saving using joblib
7. Prediction pipeline
8. Streamlit web app for user interaction

## 📊 Evaluation Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- R² Score

## 🌐 Streamlit Web App
A user-friendly Streamlit application allows users to enter house details and get real-time price predictions using the trained Linear Regression model.

## 🚀 Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- VS Code

## ▶️ How to Run
```bash
python train.py
streamlit run app.py
