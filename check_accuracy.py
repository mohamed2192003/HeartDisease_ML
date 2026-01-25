import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


file = r"F:\Graduation Project\Datasets\Graduation_Project_Heart_Disease_Extended.xlsx"
data = pd.read_excel(file)

data = data.drop_duplicates()
data = data.dropna()

print("Dataset shape:", data.shape)


model = joblib.load(r"F:\Graduation Project\Code\Medical\best_heart_disease_model.pkl")
scaler = joblib.load(r"F:\Graduation Project\Code\Medical\scaler.pkl")
features = joblib.load(r"F:\Graduation Project\Code\Medical\features_columns.pkl")


x = data.drop(columns=["heart_disease"])
y = data["heart_disease"]

if "date" in x.columns:
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["date_year"] = x["date"].dt.year
    x["date_month"] = x["date"].dt.month
    x["date_day"] = x["date"].dt.day
    x["date_hour"] = x["date"].dt.hour
    x = x.drop(columns=["date"])

# One-hot encoding
x = pd.get_dummies(x, drop_first=True)

# Add missing columns
for col in features:
    if col not in x.columns:
        x[col] = 0

# Keep same order
x = x[features]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
x_train[num_cols] = scaler.transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])


y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))


plt.figure(figsize=(5, 3))
sns.countplot(x=y)
plt.title("Target Distribution (heart_disease)")
plt.xlabel("Class (0=No Disease, 1=Disease)")
plt.ylabel("Count")
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=features)
    top20 = imp.sort_values(ascending=False).head(20)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top20.values, y=top20.index)
    plt.title("Top 20 Important Features (RandomForest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

print("\nDone")