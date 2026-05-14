import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

data_dir = "data/processed_data"

X_train = pd.read_csv(f"{data_dir}/X_train.csv")
X_test = pd.read_csv(f"{data_dir}/X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(f"{data_dir}/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(f"{data_dir}/X_test_scaled.csv", index=False)

print("Normalisation terminée. Fichiers sauvegardés dans", data_dir)