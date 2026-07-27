import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA
# Load the dataset
red = pd.read_csv(r"D:\Internships\Red Wine\winequality-red.csv", sep=",")
red["type"] = "Red"
white = pd.read_csv(r"D:\Internships\White Wine\winequality-white.csv", sep=",")
white["type"] = "White"
df = pd.concat([red, white], ignore_index=True)
#1.Initial Inspection of the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print(df["quality"].value_counts().sort_index())
df = df.drop_duplicates()
#2.Excluding quality
features = df.drop(columns=["quality"])
if "type" in features.columns:
    features = pd.get_dummies(features, drop_first=True)
quality = df["quality"]
#3.Exploratory Data Analysis (EDA)
#Histogram
df.hist(figsize=(16,12))
plt.tight_layout()
plt.show()
#Boxplots
plt.figure(figsize=(14,8))
sns.boxplot(data=df.drop(columns="quality"))
plt.xticks(rotation=90)
plt.show()
#Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
#Quality distribution
sns.countplot(x="quality", data=df)
plt.show()
#Red vs White Comparison
if "type" in df.columns:

    plt.figure(figsize=(12,6))

    sns.boxplot(x="type",
                y="alcohol",
                data=df)

    plt.show()
#4.Scale Features
scaler = StandardScaler()
X = scaler.fit_transform(features)
#Train K-Means
inertia = []

silhouette = []

db_index = []

for k in range(2,9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertia.append(model.inertia_)

    silhouette.append(
        silhouette_score(X, labels)
    )

    db_index.append(
        davies_bouldin_score(X, labels)
    )
#Elbow
plt.plot(range(2,9), inertia, marker="o")

plt.xlabel("k")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()
##Sihouette
plt.plot(range(2,9), silhouette, marker="o")

plt.xlabel("k")

plt.ylabel("Silhouette")

plt.show()
#Davies-Bouldin
plt.plot(range(2,9), db_index, marker="o")

plt.xlabel("k")

plt.ylabel("Davies-Bouldin")

plt.show()
#Stability
plt.figure(figsize=(7,5))
plt.plot(k_values, stability_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Average Adjusted Rand Index")
plt.title("Cluster Stability")
plt.grid(True)
plt.show()

print("Stability Scores:")
for k, score in zip(k_values, stability_scores):
    print(f"K={k}: {score:.4f}")
#Final Model with Optimal K
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)
#5.Cluster Profiles
cluster_profile = df.groupby("Cluster")[[
    "fixed acidity",
    "residual sugar",
    "sulphates",
    "alcohol",
    "density",
    "quality"
]].mean()

print(cluster_profile)
#Cluster Quality Distribution
sns.boxplot(
    x="Cluster",
    y="quality",
    data=df
)

plt.show()
#PCA(Princial Component Analysis) for Visualization
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("Wine Clusters")

plt.show()

#6.Train Isolation Forest
contamination_levels = [
    0.01,
    0.03,
    0.05
]

for c in contamination_levels:

    iso = IsolationForest(
        contamination=c,
        random_state=42
    )

    preds = iso.fit_predict(X)

    scores = iso.decision_function(X)

    print(
        c,
        np.sum(preds==-1)
    )
 #Final Isolation Forest Model
    iso = IsolationForest(
    contamination=0.03,
    random_state=42
)

df["Anomaly"] = iso.fit_predict(X)

df["Score"] = iso.decision_function(X)
#Visualize Anomalies
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Anomaly"],
    cmap="coolwarm"
)

plt.title("Isolation Forest Anomalies")

plt.show()
#7.Inspect Anomalies
anomalies = df[df["Anomaly"]==-1]

print(anomalies.head())

print(anomalies.describe())
#8.Cluster vs Anomaly
pd.crosstab(
    df["Cluster"],
    df["Anomaly"]
)
#9.DashBoard
fig, ax = plt.subplots(2,2, figsize=(14,10))

# Cluster Count
sns.countplot(
    x="Cluster",
    data=df,
    ax=ax[0,0]
)

# Quality by Cluster
sns.boxplot(
    x="Cluster",
    y="quality",
    data=df,
    ax=ax[0,1]
)

# Anomaly Distribution
sns.countplot(
    x="Anomaly",
    data=df,
    ax=ax[1,0]
)

# PCA Plot
scatter = ax[1,1].scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.tight_layout()

plt.show()