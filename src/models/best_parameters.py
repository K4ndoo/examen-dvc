from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os

data_dir = "data/processed_data"
model_dir = "models"

X_train = pd.read_csv(f"{data_dir}/X_train_scaled.csv")
y_train = pd.read_csv(f"{data_dir}/y_train.csv")

model = RandomForestRegressor(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [15, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train.values.ravel())

best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)

os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_params, f"{model_dir}/best_params.pkl")

print(f"Meilleurs paramètres sauvegardés dans {model_dir}/best_params.pkl")