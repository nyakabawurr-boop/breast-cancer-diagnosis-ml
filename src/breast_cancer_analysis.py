import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------
DATA_PATH = "../data/breast-cancer-wisconsin-data-Decision-Tree.csv"
VISUALS_PATH = "../visuals"

os.makedirs(VISUALS_PATH, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. Basic inspection
# --------------------------------------------------
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# --------------------------------------------------
# 4. Drop unnecessary columns
# --------------------------------------------------
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "Unnamed: 32" in df.columns:
    df = df.drop(columns=["Unnamed: 32"])

print("\nShape after dropping unnecessary columns:")
print(df.shape)

# --------------------------------------------------
# 5. Target distribution
# --------------------------------------------------
print("\nDiagnosis value counts:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis percentages:")
print(df["diagnosis"].value_counts(normalize=True) * 100)

# --------------------------------------------------
# 6. Set plot style
# --------------------------------------------------
sns.set_style("whitegrid")

# --------------------------------------------------
# 7. Figure 1: Diagnosis count plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="diagnosis")
plt.title("Diagnosis Counts (B vs M)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_1_Diagnosis_Counts (B vs M).png"))
plt.show()

# --------------------------------------------------
# 8. Figure 2: Correlation heatmap
# --------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(16, 12))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_2_Feature_Correlation_Heatmap.png"))
plt.show()

# --------------------------------------------------
# 9. Boxplots for important features
# --------------------------------------------------
boxplot_features = [
    ("perimeter_worst", "Figure_3_Perimeter_Worst_by_Diagnosis.png", "Perimeter Worst by Diagnosis"),
    ("radius_worst", "Figure_4_Radius_Worst_by_Diagnosis.png", "Radius Worst by Diagnosis"),
    ("area_worst", "Figure_5_Area_Worst_by_Diagnosis.png", "Area Worst by Diagnosis"),
]

for feature, filename, title in boxplot_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="diagnosis", y=feature)
        plt.title(title)
        plt.xlabel("Diagnosis")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_PATH, filename))
        plt.show()

# --------------------------------------------------
# 10. Prepare features and target
# --------------------------------------------------
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # B=0, M=1

print("\nEncoded target classes:")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# --------------------------------------------------
# 11. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=1234,
    stratify=y_encoded
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# --------------------------------------------------
# 12. Scale features
# --------------------------------------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 13. Train Decision Tree
# --------------------------------------------------
dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=1234,
    class_weight="balanced"
)
dt_model.fit(X_train_scaled, y_train)

dt_pred = dt_model.predict(X_test_scaled)
dt_prob = dt_model.predict_proba(X_test_scaled)[:, 1]

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred, target_names=label_encoder.classes_))

# --------------------------------------------------
# 14. Decision Tree Confusion Matrix
# --------------------------------------------------
dt_cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_6_Confusion_Matrix_Decision_Tree.png"))
plt.show()

# --------------------------------------------------
# 15. Decision Tree ROC Curve
# --------------------------------------------------
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
dt_auc = roc_auc_score(y_test, dt_prob)

plt.figure(figsize=(7, 5))
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree AUC = {dt_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_7_ROC_Curve_Decision_Tree.png"))
plt.show()

# --------------------------------------------------
# 16. Decision Tree Precision-Recall Curve
# --------------------------------------------------
dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_prob)
dt_pr_auc = average_precision_score(y_test, dt_prob)

plt.figure(figsize=(7, 5))
plt.plot(dt_recall, dt_precision, label=f"Decision Tree PR-AUC = {dt_pr_auc:.3f}")
plt.title("Precision-Recall Curve - Decision Tree")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_8_Precision_Call_Curve_Decision_Tree.png"))
plt.show()

# --------------------------------------------------
# 17. Train Random Forest
# --------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=1234,
    n_jobs=-1,
    class_weight="balanced"
)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# --------------------------------------------------
# 18. Random Forest Confusion Matrix
# --------------------------------------------------
rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_9_Confusion_Matrix_Random_Forest.png"))
plt.show()

# --------------------------------------------------
# 19. Random Forest ROC Curve
# --------------------------------------------------
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
rf_auc = roc_auc_score(y_test, rf_prob)

plt.figure(figsize=(7, 5))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest AUC = {rf_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_10_ROC_Curve_Random_Forest.png"))
plt.show()

# --------------------------------------------------
# 20. Random Forest Precision-Recall Curve
# --------------------------------------------------
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
rf_pr_auc = average_precision_score(y_test, rf_prob)

plt.figure(figsize=(7, 5))
plt.plot(rf_recall, rf_precision, label=f"Random Forest PR-AUC = {rf_pr_auc:.3f}")
plt.title("Precision-Recall Curve - Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_11_Precision_Call_Curve_Random_Forest.png"))
plt.show()

# --------------------------------------------------
# 21. Random Forest Feature Importance
# --------------------------------------------------
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False).head(10)

print("\nTop 10 Feature Importances:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_PATH, "Figure_12_Feature_Importance_Random_Forest.png"))
plt.show()

# --------------------------------------------------
# 22. Final comparison summary
# --------------------------------------------------
print("\nModel Performance Summary")
print("-" * 40)
print(f"Decision Tree ROC-AUC: {dt_auc:.4f}")
print(f"Decision Tree PR-AUC : {dt_pr_auc:.4f}")
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
print(f"Random Forest PR-AUC : {rf_pr_auc:.4f}")