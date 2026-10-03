import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# 1. LOAD LABELLED DATA
# ============================================

df = pd.read_csv("data/traffic_labelled.csv")

print("Dataset shape:", df.shape)

# ============================================
# 2. SELECT FEATURES AND TARGET
# ============================================

X = df[
    [
        "way",
        "day_of_week",
        "hour"
    ]
]

y = df["congestion_level"]

print("\nFeatures:")
print(X.head())

print("\nClasses:")
print(sorted(y.unique()))

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================
# 4. FEATURE PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "road",
            OneHotEncoder(handle_unknown="ignore"),
            ["way"]
        ),
        (
            "day",
            OneHotEncoder(handle_unknown="ignore"),
            ["day_of_week"]
        ),
        (
            "hour",
            StandardScaler(),
            ["hour"]
        )
    ]
)

# ============================================
# 5. CREATE IMPROVED NEURAL NETWORK
# ============================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "neural_network",
            MLPClassifier(
                hidden_layer_sizes=(20, 10),
                activation="logistic",
                solver="lbfgs",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# ============================================
# 6. TRAIN MODEL
# ============================================

print("\nTraining improved neural network...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")

# ============================================
# 7. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# 8. CALCULATE ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n============================================")
print("IMPROVED MODEL ACCURACY")
print("============================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ============================================
# 9. CLASSIFICATION REPORT
# ============================================

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================
# 10. CONFUSION MATRIX
# ============================================

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# ============================================
# 11. CONFUSION MATRIX HEATMAP
# ============================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[0, 1, 2, 3, 4],
    yticklabels=[0, 1, 2, 3, 4]
)

plt.title("Confusion Matrix - Improved Model")

plt.xlabel(
    "Predicted Congestion Level"
)

plt.ylabel(
    "Actual Congestion Level"
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved to "
    "results/confusion_matrix.png"
)

# ============================================
# 12. BASELINE VS IMPROVED ACCURACY
# ============================================

baseline_accuracy = 39.07
improved_accuracy = accuracy * 100

models = [
    "Baseline",
    "Improved"
]

accuracies = [
    baseline_accuracy,
    improved_accuracy
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.title("Baseline vs Improved Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.show()

print(
    "\nAccuracy comparison saved to "
    "results/accuracy_comparison.png"
)

# ============================================
# 13. SAVE FINAL MODEL
# ============================================

import joblib

joblib.dump(
    model,
    "results/traffic_congestion_model.joblib"
)

print(
    "Final model saved to "
    "results/traffic_congestion_model.joblib"
)

# ============================================
# 14. FINAL SUMMARY
# ============================================

print("\n============================================")
print("PROJECT MODEL SUMMARY")
print("============================================")

print(
    "Baseline Accuracy:",
    f"{baseline_accuracy:.2f}%"
)

print(
    "Improved Accuracy:",
    f"{improved_accuracy:.2f}%"
)

print(
    "Improvement:",
    f"{improved_accuracy - baseline_accuracy:.2f}",
    "percentage points"
)