import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = {
    "age": [22, 35, 58, 29, 41, 25, 60, 38, 44, 27,
            32, 46, 52, 24, 31, 48, 36, 40, 59, 28],
    "monthly_income": [25000, 60000, 85000, 35000, 72000, 27000, 95000, 64000, 77000, 30000,
                       55000, 82000, 73000, 26000, 48000, 91000, 62000, 70000, 88000, 34000],
    "monthly_savings": [5000, 10000, 15000, 6000, 13000, 4000, 20000, 11000, 14000, 5500,
                        9000, 14000, 12000, 3500, 8000, 17000, 10000, 11500, 16000, 5000],
    "investment_experience": [1, 5, 10, 2, 7, 1, 15, 6, 9, 0,
                              4, 11, 8, 1, 3, 12, 5, 6, 14, 2],
    "investment_time_horizon": [2, 5, 10, 4, 7, 3, 15, 6, 12, 5,
                                6, 12, 10, 3, 5, 13, 7, 8, 14, 4]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['risk_category'] = clusters
df['cluster'] = kmeans.fit_predict(X_scaled)


centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_original, columns=df.columns[:5])
centroid_df['cluster'] = [0, 1, 2]
print(centroid_df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.7)
plt.title("K-Means Clusters (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

centers = pca.transform(kmeans.cluster_centers_)

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

df['risk_category_label'] = df['risk_category'].map({
    0: "Medium Risk",
    1: "Low Risk",
    2: "High Risk"
})

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(df)