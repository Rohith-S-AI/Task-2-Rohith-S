# Task-2-Rohith-S
Repository for task 2
# 🌸 Data Classification Using AI | DecodeLabs Project 2

> **AI Industrial Training — Batch 2026 | Powered by DecodeLabs**

---

## 📌 About the Project

A supervised machine learning classification model built using Python and Scikit-learn.  
This project demonstrates the full ML pipeline — from raw data to intelligent predictions — using the **Iris Benchmark Dataset** and the **K-Nearest Neighbors (KNN)** algorithm.

---

## 🧠 Concept

Project 1 taught machines to follow rules we wrote.  
Project 2 teaches machines to **derive the rules themselves from data.**

> *"We do not write the rules. We provide history, and the machine derives the logic."*

---

## ⚙️ Full Pipeline (IPO Framework)

```
INPUT               →     PROCESS            →     OUTPUT
Iris Dataset             Train-Test Split          Confusion Matrix
Feature Scaling          KNN Algorithm (K=5)       F1 Score
```

---

## 🗃️ Dataset — The Iris Benchmark

| Property | Value |
|----------|-------|
| Samples | 150 (Balanced) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Features | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |

---

## 🚀 Features

- ✅ Load and explore the Iris dataset
- ✅ Train-Test Split (80/20) with shuffle to remove order bias
- ✅ Feature Scaling via StandardScaler (Mean=0, Variance=1)
- ✅ KNN Classification with optimal K=5
- ✅ Confusion Matrix output
- ✅ Precision, Recall, F1 Score report
- ✅ Live prediction on new unseen data

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Scikit-learn | ML pipeline (KNN, Scaler, Metrics) |
| NumPy | Numerical operations |

---

## ▶️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Rohith-S-AI/Task-2-Rohith S.git

# Navigate into folder
Classifier.py

# Install dependencies
pip install scikit-learn numpy

# Run the classifier
python classifier.py
```

---

## 📊 Sample Output

```
📦 Dataset Loaded: Iris Benchmark
   Samples   : 150 | Features : 4 | Classes : 3

✂️  Train-Test Split:
   Training : 120 samples | Testing : 30 samples

⚙️  Feature Scaling Applied: StandardScaler

🤖 Model: KNeighborsClassifier | K=5

📊 Confusion Matrix:
        Setosa  Versicolor  Virginica
  setosa       [10  0  0]
  versicolor   [ 0  9  0]
  virginica    [ 0  0 11]

✅ Weighted F1 Score : 1.0000
✅ Accuracy          : 100.00%

   Input  : Sepal=5.1cm x 3.5cm | Petal=1.4cm x 0.2cm
   Result : SETOSA 🌸
```

---

## 📚 Key Learnings

- **Supervised Learning** — machine learns from labeled historical data
- **StandardScaler** — essential for KNN; raw data creates distance bias
- **Train-Test Split** — never test on training data (data leakage = wrong results)
- **K=5 (The Elbow)** — K=1 overfits, K=100 underfits; the elbow is optimal
- **F1 Score over Accuracy** — accuracy is a lie on imbalanced data; F1 tells the truth
- **Confusion Matrix** — shows exactly where the model gets confused

---

## 🗂️ Project Structure

```
decodelabs-ai-project2/
│
├── classifier.py    # Main ML pipeline script
└── README.md        # Project documentation
```

---

## 👤 Author

**Rohith S**  
ECE Final Year | Jeppiaar Engineering College, Chennai  
Aspiring AI Engineer | DecodeLabs Intern — Batch 2026

---

## 🏢 About DecodeLabs

Industrial AI Training Program focused on building real-world AI projects from foundations to deployment.  
🌐 [www.decodelabs.tech](http://www.decodelabs.tech)
