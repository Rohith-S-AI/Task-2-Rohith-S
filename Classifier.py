# ============================================================
# DecodeLabs - AI Industrial Training | Batch 2026
# Project 2: Data Classification Using AI
# Algorithm: K-Nearest Neighbors | Dataset: Iris
# Author: Rohi
# ============================================================

# ── IMPORTS ─────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# ── PHASE 1: INPUT — Load & Understand Dataset ───────────────
print("=" * 55)
print("   DecodeLabs | Project 2: Data Classification")
print("=" * 55)

iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

print(f"\n📦 Dataset Loaded: Iris Benchmark")
print(f"   Samples   : {X.shape[0]}")
print(f"   Features  : {X.shape[1]} (Sepal L, Sepal W, Petal L, Petal W)")
print(f"   Classes   : {len(iris.target_names)} {list(iris.target_names)}")

# ── PHASE 2: PROCESS ─────────────────────────────────────────

# Step 1 — Train-Test Split (80/20) with shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"\n✂️  Train-Test Split:")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# Step 2 — Feature Scaling (StandardScaler: Mean=0, Variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test  = scaler.transform(X_test)       # Only transform test (no data leakage)
print(f"\n⚙️  Feature Scaling Applied: StandardScaler (Mean=0, Variance=1)")

# Step 3 — KNN Model (K=5, The Elbow / Optimal K)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)              # FIT: Memorize the map
predictions = model.predict(X_test)     # PREDICT: Apply logic
print(f"\n🤖 Model: KNeighborsClassifier | K=5")

# ── PHASE 3: OUTPUT — Validation ─────────────────────────────
print("\n" + "=" * 55)
print("   OUTPUT VALIDATION")
print("=" * 55)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print(f"\n📊 Confusion Matrix:")
print(f"        Setosa  Versicolor  Virginica")
for i, row in enumerate(cm):
    print(f"  {iris.target_names[i]:12s} {row}")

# Classification Report (Precision, Recall, F1)
print(f"\n📈 Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# F1 Score (Weighted)
f1 = f1_score(y_test, predictions, average='weighted')
print(f"✅ Weighted F1 Score : {f1:.4f}")
print(f"✅ Accuracy          : {model.score(X_test, y_test) * 100:.2f}%")

# ── BONUS: Predict a new sample ──────────────────────────────
print("\n" + "=" * 55)
print("   LIVE PREDICTION TEST")
print("=" * 55)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Typical Setosa values
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
print(f"\n   Input  : Sepal=5.1cm x 3.5cm | Petal=1.4cm x 0.2cm")
print(f"   Result : {iris.target_names[result[0]].upper()} 🌸")
print("\n" + "=" * 55)
