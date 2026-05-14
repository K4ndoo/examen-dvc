import pandas as pd
import os
from sklearn.model_selection import train_test_split

input_file = "data/raw_data/raw.csv"
output_dir = "data/processed_data"

df = pd.read_csv(input_file)

X = df.drop(["silica_concentrate", "date"], axis=1)
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)