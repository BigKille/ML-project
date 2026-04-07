import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# 1. Load the dataset
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
features = iris['feature_names']

# 2. Calculate Mutual Information (MI) scores
# mutual_info_classif estimates MI for discrete target variables
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.Series(mi_scores, index=features)

# 3. Print and visualize the scores
print("Mutual Information Scores:")
print(mi_df.sort_values(ascending=False))

# Visualize the feature importance
mi_df.sort_values(ascending=False).plot.bar(figsize=(10, 5))
plt.title("Feature Importance using Mutual Information")
plt.ylabel("MI Score (nats)") # Scikit-learn uses nats by default
plt.show()