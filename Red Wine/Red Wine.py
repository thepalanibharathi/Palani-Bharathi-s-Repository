import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv(r"D:\Internships\Red Wine\winequality-red.csv", sep=",")
# First five rows
print(df.head())
# Dataset information
print(df.info())
# Statistical summary
print(df.describe())
# Quality distribution
print(df['quality'].value_counts().sort_index())

#2.Binarize the quality variable
df['quality_label'] = np.where(df['quality'] >= 7, 1, 0)

print(df['quality_label'].value_counts())

#Exploratory Data Analysis (EDA)
plt.figure(figsize=(5,4))

sns.countplot(x='quality_label', data=df)

plt.xticks([0,1], ['Bad','Good'])

plt.title("Wine Quality Class Distribution")

plt.show()

#Quality
plt.figure(figsize=(7,5))
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.show()


df['quality'] = np.where(df['quality'] >= 7, 1, 0)
print(df['quality'].value_counts())
#Class Plotting
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df)
plt.xticks([0,1],["Bad","Good"])
plt.title("Class Balance")
plt.show()

#3.EDA
df.hist(figsize=(15,12), bins=20)
plt.tight_layout()
plt.show()
#Heatmap
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Heatmap")
plt.show()

#4.Scale features using StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features (X) and Target (y)
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Display the shapes of the datasets
print("Training Features Shape:", X_train_scaled.shape)
print("Testing Features Shape:", X_test_scaled.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

#5. Decision Tree Classifier
dt = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
plt.figure(figsize=(18,10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Bad','Good'],
    filled=True,
    rounded=True
)
plt.show()

#6. K-Nearest Neighbors Classifier
k_values = [3,5,7]

best_accuracy = 0
best_k = None
best_model = None

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)

    print(f"K={k} Accuracy={acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k
        best_model = knn

print("\nBest K =", best_k)
print("Best Accuracy =", best_accuracy)

knn_pred = best_model.predict(X_test_scaled)

#7.Confusion Matrix
cm_dt = confusion_matrix(y_test, dt_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=['Bad','Good']
).plot()

plt.title("Decision Tree Confusion Matrix")

plt.show()

cm_knn = confusion_matrix(y_test, knn_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=['Bad','Good']
).plot()

plt.title("KNN Confusion Matrix")

plt.show()

#8.Decision Tree Classification Report
print("Decision Tree Classification Report")
print(classification_report(y_test, dt_pred))
#KNN Classification Report
print("KNN Classification Report")
print(classification_report(y_test, knn_pred))

#Feature Importance (Decision Tree)
importance = pd.Series(
    dt.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)
plt.figure(figsize=(8,6))
importance.plot(kind='bar')
plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")
plt.show()

#9.Comparison of Models
comparison = pd.DataFrame({
    'Model':['Decision Tree','KNN'],
    'Accuracy':[dt_accuracy,best_accuracy]
})
print(comparison)
#Accuracy Comparison Plot
plt.figure(figsize=(6,4))
sns.barplot(data=comparison, x='Model', y='Accuracy')
plt.ylim(0,1)
plt.title("Model Comparison")
plt.show()