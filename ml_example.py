import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize the data
plt.figure(figsize=(12, 5))

# Plot 1: Scatter plot of the two most important features
plt.subplot(1, 2, 1)
for i, species in enumerate(iris.target_names):
    plt.scatter(df[df['target'] == i]['petal length (cm)'], 
                df[df['target'] == i]['petal width (cm)'], 
                label=species, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Species Classification')
plt.legend()

# Plot 2: Feature importance
plt.subplot(1, 2, 2)
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Random Forest Model')

plt.tight_layout()
plt.show()

# Make a prediction for a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example measurements
prediction = model.predict(new_sample)
probability = model.predict_proba(new_sample)

print(f"\nPrediction for new sample {new_sample[0]}:")
print(f"Predicted class: {iris.target_names[prediction][0]}")
print(f"Prediction probabilities: {dict(zip(iris.target_names, probability[0]))}")