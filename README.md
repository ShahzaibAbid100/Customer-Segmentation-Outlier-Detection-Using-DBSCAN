# Customer Segmentation & Outlier Detection Using DBSCAN

## Project Overview

This project applies **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to segment customers based on their purchasing characteristics.

DBSCAN is an unsupervised machine learning algorithm that identifies clusters based on dense regions of data. Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand and can also identify **noise and outliers**.

The project focuses on understanding DBSCAN, selecting suitable `eps` and `min_samples` values, identifying customer groups, detecting noise points, and evaluating the discovered clusters.

---

## Objective

The main objectives of this project are:

- Apply DBSCAN clustering to customer data
- Understand density-based clustering
- Standardize features before clustering
- Experiment with `eps` and `min_samples`
- Identify customer clusters
- Detect noise and outliers
- Visualize discovered customer groups
- Evaluate clusters using Silhouette Score
- Analyze the characteristics of each cluster

---

## Dataset

The project uses the **Mall Customers Dataset**.

Important columns include:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

For this project, the main clustering features are:

- Annual Income
- Spending Score

Since DBSCAN is an **unsupervised learning algorithm**, there is no target variable `y`.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- DBSCAN
- StandardScaler
- NearestNeighbors
- Silhouette Score

---

## Project Workflow

```text
Load Dataset
      ↓
Explore Data
      ↓
Check Missing Values
      ↓
Check Duplicates
      ↓
Select Features
      ↓
Feature Scaling
      ↓
K-Distance Plot
      ↓
Choose eps & min_samples
      ↓
Train DBSCAN
      ↓
Assign Cluster Labels
      ↓
Detect Noise (-1)
      ↓
Visualize Clusters
      ↓
Silhouette Score
      ↓
Analyze Customer Groups
```

---

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
```

---

## 2. Load Dataset

```python
df = pd.read_csv("Mall_Customers.csv")

df.head()
```

---

## 3. Explore the Dataset

Check dataset dimensions:

```python
print(df.shape)
```

Check column information:

```python
df.info()
```

View statistical information:

```python
df.describe()
```

Check column names:

```python
print(df.columns)
```

---

## 4. Check Missing Values

```python
print(df.isnull().sum())
```

Missing values should be handled before applying DBSCAN.

---

## 5. Check Duplicate Records

```python
print(df.duplicated().sum())
```

If duplicates need to be removed:

```python
df = df.drop_duplicates()
```

---

## 6. Select Features

For this project, Annual Income and Spending Score are used:

```python
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]
```

There is no target variable because DBSCAN is an unsupervised learning algorithm.

```text
Supervised Learning:

X → Features
y → Target


DBSCAN:

X → Features
No y
```

---

## 7. Visualize Data Before Clustering

```python
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"]
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customers Before DBSCAN Clustering")

plt.show()
```

This allows us to inspect the distribution of customers before applying the clustering algorithm.

---

## 8. Feature Scaling

DBSCAN uses distances between data points, so feature scaling is important.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

StandardScaler places the selected features on comparable scales.

Without scaling, a feature with larger numerical values could have a greater influence on distance calculations.

---

## 9. K-Distance Plot

One of the most important DBSCAN hyperparameters is `eps`.

A K-distance plot can help investigate a suitable value.

```python
neighbors = NearestNeighbors(
    n_neighbors=5
)

neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(
    X_scaled
)
```

Select the distance to the fifth nearest neighbor:

```python
distances = np.sort(
    distances[:, 4]
)
```

Plot the distances:

```python
plt.plot(distances)

plt.xlabel("Data Points")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("K-Distance Plot")

plt.show()
```

A sharp bend or knee in this graph can provide a useful candidate for the `eps` value.

---

## 10. Create the DBSCAN Model

Example:

```python
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)
```

The two main parameters are:

### `eps`

`eps` defines the maximum neighborhood distance used when determining nearby points.

### `min_samples`

`min_samples` specifies the minimum number of samples required in a neighborhood for a point to qualify as a core point.

---

## 11. Train DBSCAN

```python
clusters = dbscan.fit_predict(
    X_scaled
)
```

DBSCAN returns cluster labels such as:

```text
0
1
2
-1
```

where:

```text
0, 1, 2 ... → Cluster labels
-1          → Noise / Outlier
```

---

## 12. Add Cluster Labels

```python
df["Cluster"] = clusters

df.head()
```

Each customer now has a cluster label.

---

## 13. Check Cluster Distribution

```python
print(
    df["Cluster"].value_counts()
)
```

This shows the number of customers assigned to each cluster as well as the number classified as noise.

---

## 14. Count Number of Clusters

```python
n_clusters = len(
    set(clusters)
) - (1 if -1 in clusters else 0)

print(
    "Number of clusters:",
    n_clusters
)
```

The `-1` label is excluded because it represents noise rather than an actual cluster.

---

## 15. Count Noise Points

```python
n_noise = list(
    clusters
).count(-1)

print(
    "Number of noise points:",
    n_noise
)
```

Calculate the percentage of noise:

```python
noise_percentage = (
    n_noise / len(clusters)
) * 100

print(
    "Noise Percentage:",
    noise_percentage
)
```

---

## 16. Visualize DBSCAN Clusters

```python
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=clusters
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using DBSCAN")

plt.show()
```

The visualization shows the customer groups identified by DBSCAN.

Noise observations have the cluster label `-1`.

---

## 17. Experiment With Different `eps` Values

Different `eps` values can produce very different clustering results.

```python
eps_values = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.6
]

for eps in eps_values:

    model = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels = model.fit_predict(
        X_scaled
    )

    n_clusters = len(
        set(labels)
    ) - (1 if -1 in labels else 0)

    n_noise = list(
        labels
    ).count(-1)

    print(
        f"eps={eps}, "
        f"clusters={n_clusters}, "
        f"noise={n_noise}"
    )
```

Generally:

```text
Small eps
    ↓
Smaller neighborhoods
    ↓
More observations may become noise


Large eps
    ↓
Larger neighborhoods
    ↓
More observations become connected
    ↓
Separate clusters may merge
```

---

## 18. Experiment With `min_samples`

```python
min_samples_values = [
    3,
    4,
    5,
    6,
    8,
    10
]

for samples in min_samples_values:

    model = DBSCAN(
        eps=0.4,
        min_samples=samples
    )

    labels = model.fit_predict(
        X_scaled
    )

    n_clusters = len(
        set(labels)
    ) - (1 if -1 in labels else 0)

    n_noise = list(
        labels
    ).count(-1)

    print(
        f"min_samples={samples}, "
        f"clusters={n_clusters}, "
        f"noise={n_noise}"
    )
```

Increasing `min_samples` requires a denser neighborhood for points to qualify as core points.

---

## 19. Remove Noise for Cluster Evaluation

DBSCAN uses `-1` for noise.

A Boolean mask can be created:

```python
mask = clusters != -1
```

Keep only non-noise observations:

```python
clean_X = X_scaled[mask]
```

Keep their corresponding cluster labels:

```python
clean_clusters = clusters[mask]
```

This gives:

```text
Original data
     ↓
Remove points with label -1
     ↓
clean_X
clean_clusters
```

---

## 20. Silhouette Score

If DBSCAN discovers at least two non-noise clusters, Silhouette Score can be calculated:

```python
if len(set(clean_clusters)) >= 2:

    score = silhouette_score(
        clean_X,
        clean_clusters
    )

    print(
        "Silhouette Score:",
        score
    )
```

Silhouette Score ranges approximately from `-1` to `1`.

```text
Closer to 1
→ Clusters are relatively well separated

Around 0
→ Clusters overlap

Below 0
→ Some observations may fit another
  cluster better
```

Silhouette Score should not be the only criterion used when selecting DBSCAN parameters.

The number of clusters, number of noise points, visualization, and practical interpretation should also be considered.

---

## 21. Identify Outliers

Customers identified as noise can be extracted using:

```python
outliers = df[
    df["Cluster"] == -1
]

print(outliers)
```

Count them:

```python
print(
    "Total Outliers:",
    len(outliers)
)
```

This is one of the important advantages of DBSCAN: it can identify observations that do not belong to sufficiently dense regions.

---

## 22. Analyze the Clusters

Remove noise:

```python
clustered_data = df[
    df["Cluster"] != -1
]
```

Calculate average characteristics:

```python
cluster_summary = clustered_data.groupby(
    "Cluster"
)[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print(cluster_summary)
```

These averages can be used to understand the characteristics of the customer groups discovered by DBSCAN.

Cluster descriptions should be based on the actual results rather than the numeric cluster labels.

---

## How DBSCAN Works

DBSCAN groups observations based on density.

The basic process is:

```text
Select a data point
        ↓
Find points within eps
        ↓
Are there enough neighbors?
        ↓
      Yes
        ↓
Core Point
        ↓
Expand the dense region
        ↓
Form Cluster
```

Points that are not sufficiently connected to a dense region may be classified as:

```text
Noise → -1
```

---

## Types of Points in DBSCAN

### Core Point

A point with enough observations in its `eps` neighborhood.

### Border Point

A point that is not itself a core point but is reachable from a core point and belongs to the cluster.

### Noise Point

A point that is not sufficiently connected to a dense region.

In scikit-learn, noise is represented by:

```text
-1
```

---

## DBSCAN vs K-Means

| K-Means | DBSCAN |
|---|---|
| Requires number of clusters | Does not require number of clusters beforehand |
| Centroid-based | Density-based |
| Assigns every observation to a cluster | Can identify noise |
| Uses `n_clusters` | Uses `eps` and `min_samples` |
| Often suited to compact cluster structures | Can identify some irregularly shaped dense clusters |

---

## DBSCAN vs Hierarchical Clustering

| Hierarchical Clustering | DBSCAN |
|---|---|
| Builds a hierarchy of clusters | Finds density-based clusters |
| Can be visualized with a dendrogram | Does not use a dendrogram |
| Uses linkage methods | Uses `eps` and `min_samples` |
| Typically assigns observations to clusters after cutting/selecting hierarchy | Can classify observations as noise |

---

## Key Concepts Learned

Through this project, I practiced:

- Unsupervised Machine Learning
- DBSCAN Clustering
- Density-Based Clustering
- Customer Segmentation
- Feature Scaling
- StandardScaler
- `eps`
- `min_samples`
- Core Points
- Border Points
- Noise Points
- Outlier Detection
- K-Distance Plot
- Boolean Masking
- Silhouette Score
- Cluster Analysis
- Data Visualization

---

## Project Results

After running the final model, add your actual results here.

```text
Selected eps: [Your eps]

Selected min_samples: [Your value]

Number of clusters: [Your result]

Number of noise points: [Your result]

Noise percentage: [Your result]

Silhouette Score: [Your score]
```

### Cluster Interpretation

```text
Cluster 0:
[Describe the actual characteristics]

Cluster 1:
[Describe the actual characteristics]

Cluster 2:
[Describe the actual characteristics]

Noise / Outliers:
[Describe any noticeable characteristics]
```

Do not assume that Cluster 0, Cluster 1, etc. have a particular meaning. Interpret them using your actual cluster statistics.

---

## Future Improvements

Possible improvements include:

- Testing additional `eps` values
- Testing different `min_samples` values
- Adding additional customer features
- Comparing DBSCAN with K-Means
- Comparing DBSCAN with Hierarchical Clustering
- Performing more detailed outlier analysis
- Using PCA for multidimensional visualization
- Testing other density-based clustering approaches

---

## Conclusion

This project demonstrates how **DBSCAN** can be used for customer segmentation and outlier detection.

Unlike clustering methods that require a predefined number of clusters, DBSCAN discovers clusters based on the density of observations and can label isolated observations as noise.

The project demonstrates feature scaling, K-distance analysis, hyperparameter experimentation, cluster evaluation using Silhouette Score, and analysis of detected outliers.

---
