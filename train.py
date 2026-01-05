import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/housing.csv")

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical columns to numbers
X_encoded = pd.get_dummies(X, drop_first=True)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train LinearRegression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)


rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Performance:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)

# Save the better model
joblib.dump(model, "model.pkl")
joblib.dump(X_encoded.columns, "features.pkl")

print("Model and feature list saved successfully")


