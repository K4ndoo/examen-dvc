import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

data_dir = "data/processed_data"
model_dir = "models"
metrics_dir = "metrics"
output_data_dir = "data"

X_test = pd.read_csv(f"{data_dir}/X_test_scaled.csv")
y_test = pd.read_csv(f"{data_dir}/y_test.csv")

model = joblib.load(f"{model_dir}/final_model.pkl")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

os.makedirs(metrics_dir, exist_ok=True)
scores = {
    "mse": mse,
    "r2": r2
}

with open(f"{metrics_dir}/scores.json", "w") as f:
    json.dump(scores, f, indent=4)
print(f"Métriques sauvegardées dans {metrics_dir}/scores.json")

predictions_df = pd.DataFrame({
    "y_true": y_test.values.ravel(),
    "y_pred": y_pred
})

predictions_df.to_csv(f"{output_data_dir}/predictions.csv", index=False)
print(f"Dataset de prédictions sauvegardé dans {output_data_dir}/predictions.csv")