# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
file_path = r"C:\Users\kanik\Downloads\Darknet.csv"
df = pd.read_csv(file_path)
print(f"Dataset shape: {df.shape}")
print("\nDataset preview:")
print(df.head())
df.columns = df.columns.str.strip()
label_column_candidates = [col for col in df.columns if 'Label' in col]
if not label_column_candidates:
    raise ValueError("No label column found in the dataset.")
label_column = label_column_candidates[0]
print(f"\nUsing label column: {label_column}")

irrelevant_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True, errors="ignore")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

imputer = SimpleImputer(strategy="median")
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

label_encoders = {}
for col in categorical_columns:
    if col != label_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])

X = df.drop(columns=[label_column])
y = df[label_column]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Model Accuracy: {acc:.4f}")
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, "Random Forest")

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
print("\nTop 10 Important Features from Random Forest:")
print(feature_importances.head(10))
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, "SVM")

print("\nCybersecurity Threat Classification Completed!")
