A Machine Learning web application that predicts the estimated price of used cars in Egyptian Pounds (EGP) based on their specifications.
## 📌 Overview
This project uses an **XGBoost Regression** model to estimate used car prices from features such as brand, model, mileage, fuel type, transmission, engine capacity, body type, and manufacturing year.

The trained model is integrated into an interactive **Streamlit** web application where users can enter car specifications and receive an estimated price instantly.

## 🎯 Project Objectives


- Build a regression model for used car price prediction.
- Handle numerical and categorical car features.
- Apply appropriate preprocessing techniques.
- Train and evaluate an XGBoost regression model.
- Deploy the trained model through a user-friendly Streamlit interface.

## 🧠 Machine Learning Approach

### Data Preprocessing

The dataset was processed using several steps:

- Numerical feature preprocessing
- Categorical feature encoding using One-Hot Encoding
- Feature alignment between training and prediction data
- Creation of a `Car_Age` feature
- Log transformation of the target variable

### Model

The project uses:

**XGBoost Regressor**

The target variable was log-transformed during training to help handle the skewed distribution of car prices.

The predicted value is converted back to the original price scale using:

python
np.expm1(prediction_log)## 📊 Features

| Feature | Description |
|---|---|
| Brand | Car manufacturer |
| Model | Car model |
| Kilometers | Total mileage |
| Fuel Type | Type of fuel |
| Transmission Type | Transmission system |
| Engine Capacity (CC) | Engine capacity |
| Body Type | Car body type |
| Manufacturing Year | Production year |
| Car Age | Calculated age of the car |

## 🛠️ Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**

## 🌐 Streamlit Application

The application provides an interactive interface where users can:

1. Select the car brand and model.
2. Select the fuel and transmission types.
3. Select the body type.
4. Enter the mileage.
5. Enter the manufacturing year.
6. Enter the engine capacity.
7. Get the estimated car price in EGP.

## 📂 Project Structure

```text
used-car-price-prediction/
│
├── app.py
├── car.jpg
├── car_price_model.json
├── scaler.pkl
├── feature_columns.pkl
├── category_options.pkl
├── preprocessing_stats.pkl
├── README.md
└── .gitignore
