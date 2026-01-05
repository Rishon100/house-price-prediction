import pandas as pd
import joblib

# Load model and features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# Sample input (change values to test)
input_data = {
    "area": 7420,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# One-hot encode
df_input = pd.get_dummies(df_input)

# Align with training features
df_input = df_input.reindex(columns=features, fill_value=0)

# Predict
prediction = model.predict(df_input)

print("Estimated House Price:", int(prediction[0]))
