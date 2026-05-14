import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

data_dir = "data/processed_data"
model_dir = "models"

X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
y_train = pd.read_csv(f"{data_dir}/y_train.csv")

best_params = joblib.load(f"{model_dir}/best_params.pkl")

model = RandomForestRegressor(**best_params, random_state=42)

model.fit(X_train, y_train.values.ravel())
joblib.dump(model, f"{model_dir}/final_model.pkl")