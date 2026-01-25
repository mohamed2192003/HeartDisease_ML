import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

file = r"F:\Graduation Project\Datasets\Graduation_Project_Heart_Disease_Extended.xlsx"
data = pd.read_excel(file)

data = data.drop_duplicates()
data = data.dropna()

x = data.drop(columns=["heart_disease"])
y = data["heart_disease"]

# date features
if "date" in x.columns:
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["date_year"] = x["date"].dt.year
    x["date_month"] = x["date"].dt.month
    x["date_day"] = x["date"].dt.day
    x["date_hour"] = x["date"].dt.hour
    x = x.drop(columns=["date"])

# encoding
x = pd.get_dummies(x, drop_first=True)

# split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# scaling
scaler = StandardScaler()
num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()

x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

# SMOTE
sm = SMOTE(random_state=42)
x_train_bal, y_train_bal = sm.fit_resample(x_train, y_train)

# train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train_bal, y_train_bal)

# save
model = joblib.load(r"F:\Graduation Project\Code\Medical\best_heart_disease_model.pkl")
scaler = joblib.load(r"F:\Graduation Project\Code\Medical\scaler.pkl")
features = joblib.load(r"F:\Graduation Project\Code\Medical\features_columns.pkl")

print("✅ Model saved successfully!")