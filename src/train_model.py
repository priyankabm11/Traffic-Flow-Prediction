import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# 1. LOAD LABELLED DATA
# ============================================

df = pd.read_csv("data/traffic_labelled.csv")

print("Dataset shape:", df.shape)

# ============================================
# 2. CONVERT ROAD NAME INTO NUMERIC LABEL
# ============================================

df["road_id"] = pd.factorize(df["way"])[0]

# ============================================
# 3. SELECT INPUT FEATURES AND TARGET
# ============================================

X = df[
    [
        "road_id",
        "day_of_week",
        "hour"
    ]
]

y = df["congestion_level"]

print("\nInput features:")
print(X.head())

print("\nTarget classes:")
print(sorted(y.unique()))

# ============================================
# 4. SPLIT INTO TRAINING AND TESTING DATA
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
# 5. SCALE THE INPUT FEATURES
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. CREATE NEURAL NETWORK
# ============================================

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="logistic",
    solver="adam",
    max_iter=1000,
    random_state=42
)

# ============================================
# 7. TRAIN MODEL
# ============================================

print("\nTraining neural network...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training completed!")

# ============================================
# 8. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test_scaled)

# ============================================
# 9. CALCULATE ACCURACY
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n============================================")
print("MODEL ACCURACY")
print("============================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# ============================================
# 10. CLASSIFICATION REPORT
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
# 11. CONFUSION MATRIX
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
# 12. PLOT TRAINING LOSS
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    model.loss_curve_
)

plt.title(
    "Neural Network Training Loss"
)

plt.xlabel(
    "Iterations"
)

plt.ylabel(
    "Loss"
)

plt.grid()

plt.savefig(
    "results/training_loss.png"
)

plt.show()

print("\nTraining loss graph saved to results/training_loss.png")