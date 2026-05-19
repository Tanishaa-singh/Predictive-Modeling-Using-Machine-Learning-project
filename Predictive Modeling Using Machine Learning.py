# ==========================================
# Predictive Modeling Using Machine Learning
# Random Forest Classifier using Iris Dataset
# ==========================================

# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

print("Loading dataset...")

# Load Iris Dataset
iris = load_iris()

# Convert dataset into DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Display first 5 rows
print("\nDataset Preview:")
print(df.head())

# Features (X) and Target (y)
X = df.drop(columns=['target'])
y = df['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 2. TRAIN THE MODEL
# ==========================================

print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 4. EVALUATE THE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. VISUALIZE PERFORMANCE
# ==========================================

sns.set_style("whitegrid")

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, ax=ax)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

# Save the figure
plt.savefig("confusion_matrix.png", dpi=300)

print("\nVisualization saved as 'confusion_matrix.png'")

# Show plot
plt.show()

# ==========================================
# END OF PROJECT
# ==========================================