#!/usr/bin/env python3
"""
ULTIMATE MACHINE LEARNING SUITE
===============================

A comprehensive machine learning suite in a single file including:
- Basic ML classification
- Advanced ML with multiple algorithms
- Neural networks
- Clustering and anomaly detection
- Automated ML pipeline
- Comprehensive visualization

Author: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.datasets import load_iris, load_wine, load_boston, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Statistical Analysis
from scipy import stats
import scipy.cluster.hierarchy as sch

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# SECTION 1: BASIC ML EXAMPLE
# =============================================================================

def basic_ml_example():
    """
    Basic machine learning example with Iris dataset
    """
    print("🤖 BASIC ML EXAMPLE")
    print("=" * 50)
    
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

# =============================================================================
# SECTION 2: ADVANCED ML CLASSIFIER
# =============================================================================

class AdvancedMLClassifier:
    """
    Advanced Machine Learning Classifier with multiple algorithms and evaluation metrics
    """
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
    def load_dataset(self, dataset_name='iris'):
        """
        Load different datasets for classification
        """
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError("Dataset must be 'iris', 'wine', or 'breast_cancer'")
            
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.dataset_name = dataset_name
        
        print(f"Loaded {dataset_name} dataset")
        print(f"Features: {self.X.shape[1]}")
        print(f"Samples: {self.X.shape[0]}")
        print(f"Classes: {len(self.target_names)}")
        print(f"Class names: {list(self.target_names)}")
        
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2):
        """
        Preprocess the data: split and scale
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
        
    def train_models(self):
        """
        Train all models and find the best one
        """
        print("\nTraining models...")
        model_scores = {}
        
        for name, model in self.models.items():
            # Train the model
            if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                # These models benefit from scaled data
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                # Tree-based models don't require scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            model_scores[name] = accuracy
            self.trained_models[name] = model
            
            print(f"{name}: {accuracy:.4f}")
        
        # Find the best model
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.trained_models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name} with accuracy: {model_scores[self.best_model_name]:.4f}")
        
    def cross_validate_models(self, cv=5):
        """
        Perform cross-validation on all models
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        for name, model in self.models.items():
            if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=cv)
            else:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv)
                
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def tune_best_model(self):
        """
        Perform hyperparameter tuning on the best model
        """
        print(f"\nTuning hyperparameters for {self.best_model_name}...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)
            X_train_data = self.X_train
            
        elif self.best_model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            model = SVC(random_state=42, probability=True)
            X_train_data = self.X_train_scaled
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
            X_train_data = self.X_train
            
        else:
            print("No hyperparameter tuning defined for this model")
            return
            
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_data, self.y_train)
        
        # Update the best model
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        """
        Comprehensive evaluation of the best model
        """
        print(f"\nEvaluating {self.best_model_name}...")
        
        # Select appropriate data based on model type
        if self.best_model_name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            y_pred = self.best_model.predict(self.X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)
            X_test_data = self.X_test_scaled
        else:
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)
            X_test_data = self.X_test
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(15, 5))
        
        # Plot confusion matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.target_names, 
                    yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(1, 3, 2)
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.bar(range(len(indices)), importances[indices])
            plt.title('Top 10 Feature Importances')
            plt.xticks(range(len(indices)), 
                      [self.feature_names[i] for i in indices], 
                      rotation=45, ha='right')
        
        # ROC Curve (for binary classification)
        if len(self.target_names) == 2:
            plt.subplot(1, 3, 3)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
        else:
            # For multi-class, show a simple prediction distribution
            plt.subplot(1, 3, 3)
            unique, counts = np.unique(y_pred, return_counts=True)
            plt.bar([self.target_names[i] for i in unique], counts)
            plt.title('Prediction Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_data(self):
        """
        Visualize the dataset using PCA and t-SNE
        """
        print("\nVisualizing data...")
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X)
        
        plt.figure(figsize=(15, 6))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, cmap='viridis')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization')
        plt.colorbar(scatter, ticks=range(len(self.target_names)))
        plt.clim(-0.5, len(self.target_names)-0.5)
        
        # t-SNE plot
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y, cmap='viridis')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization')
        plt.colorbar(scatter, ticks=range(len(self.target_names)))
        plt.clim(-0.5, len(self.target_names)-0.5)
        
        # Add custom colorbar labels
        ax = plt.gca()
        cbar = ax.collections[-1].colorbar
        cbar.set_ticks(range(len(self.target_names)))
        cbar.set_ticklabels(self.target_names)
        
        plt.tight_layout()
        plt.show()
        
        # Show explained variance for PCA
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained by 2 PCs: {sum(pca.explained_variance_ratio_):.2%}")

    def predict_sample(self, sample_data):
        """
        Make a prediction for a new sample
        """
        if self.best_model_name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            sample_scaled = self.scaler.transform([sample_data])
            prediction = self.best_model.predict(sample_scaled)[0]
            probabilities = self.best_model.predict_proba(sample_scaled)[0]
        else:
            prediction = self.best_model.predict([sample_data])[0]
            probabilities = self.best_model.predict_proba([sample_data])[0]
            
        print(f"\nPrediction for sample {sample_data}:")
        print(f"Predicted class: {self.target_names[prediction]}")
        print("Class probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  {self.target_names[i]}: {prob:.4f}")
            
        return prediction, probabilities

def run_advanced_ml_example():
    """
    Run the advanced ML example
    """
    print("\n" + "=" * 60)
    print("🤖 ADVANCED MACHINE LEARNING CLASSIFIER")
    print("=" * 60)
    
    # Create classifier instance
    classifier = AdvancedMLClassifier()
    
    # Select dataset (you can change this to 'wine' or 'breast_cancer')
    dataset_name = 'iris'  # Options: 'iris', 'wine', 'breast_cancer'
    classifier.load_dataset(dataset_name)
    
    # Preprocess data
    classifier.preprocess_data()
    
    # Visualize data
    classifier.visualize_data()
    
    # Train models
    classifier.train_models()
    
    # Cross-validate models
    classifier.cross_validate_models()
    
    # Tune the best model
    classifier.tune_best_model()
    
    # Evaluate the best model
    classifier.evaluate_model()
    
    # Make a sample prediction
    if dataset_name == 'iris':
        sample = [5.1, 3.5, 1.4, 0.2]  # Example iris measurements
    elif dataset_name == 'wine':
        # Use the mean values for first 4 features as example
        sample = np.mean(classifier.X[:, :4], axis=0)
    else:  # breast_cancer
        # Use the mean values for first 4 features as example
        sample = np.mean(classifier.X[:, :4], axis=0)
        
    classifier.predict_sample(sample)
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED ANALYSIS COMPLETE")
    print("=" * 60)

# =============================================================================
# SECTION 3: NEURAL NETWORKS
# =============================================================================

def neural_network_example():
    """
    Neural network examples using TensorFlow/Keras
    """
    print("\n" + "=" * 60)
    print("🧠 NEURAL NETWORK CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Generate a synthetic dataset for binary classification
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    
    # Create the neural network model
    print("\nCreating neural network model...")
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # MULTICLASS NEURAL NETWORK EXAMPLE
    print("\n" + "=" * 60)
    print("🌸 MULTICLASS NEURAL NETWORK WITH IRIS DATASET")
    print("=" * 60)
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Iris dataset shape: {X.shape}")
    print(f"Classes: {iris.target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert labels to categorical one-hot encoding
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes=3)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes=3)
    
    # Create the neural network model for multiclass classification
    print("\nCreating multiclass neural network model...")
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        X_train_scaled, y_train_categorical,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("🎯 NEURAL NETWORK EXAMPLES COMPLETE")
    print("=" * 60)

# =============================================================================
# SECTION 4: COMPREHENSIVE ML SUITE
# =============================================================================

class ComprehensiveMLSuite:
    """
    A comprehensive machine learning suite with classification, regression,
    clustering, anomaly detection, and advanced analytics.
    """
    
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.results = {}
        
    def load_datasets(self):
        """
        Load various datasets for different ML tasks
        """
        print("📥 Loading datasets...")
        
        # Classification datasets
        iris = load_iris()
        self.datasets['iris'] = {
            'X': iris.data,
            'y': iris.target,
            'feature_names': iris.feature_names,
            'target_names': iris.target_names,
            'task': 'classification'
        }
        
        wine = load_wine()
        self.datasets['wine'] = {
            'X': wine.data,
            'y': wine.target,
            'feature_names': wine.feature_names,
            'target_names': wine.target_names,
            'task': 'classification'
        }
        
        # Regression dataset
        # Note: load_boston is deprecated, so we'll create a synthetic regression dataset
        X_reg, y_reg = make_regression(n_samples=500, n_features=13, noise=0.1, random_state=42)
        self.datasets['regression'] = {
            'X': X_reg,
            'y': y_reg,
            'feature_names': [f'Feature_{i}' for i in range(X_reg.shape[1])],
            'target_names': None,
            'task': 'regression'
        }
        
        # Synthetic datasets for clustering and anomaly detection
        X_cluster, y_cluster = make_classification(
            n_samples=300, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, n_classes=3, random_state=42
        )
        self.datasets['clustering'] = {
            'X': X_cluster,
            'y': y_cluster,
            'feature_names': ['Feature 1', 'Feature 2'],
            'target_names': ['Cluster 1', 'Cluster 2', 'Cluster 3'],
            'task': 'clustering'
        }
        
        # Anomaly detection dataset
        X_anomaly = np.random.randn(200, 2)
        # Add some anomalies
        X_anomaly = np.concatenate([X_anomaly, np.random.uniform(low=-4, high=4, size=(10, 2))])
        self.datasets['anomaly'] = {
            'X': X_anomaly,
            'y': None,
            'feature_names': ['Feature 1', 'Feature 2'],
            'target_names': None,
            'task': 'anomaly_detection'
        }
        
        print(f"✅ Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def exploratory_data_analysis(self, dataset_name='iris'):
        """
        Perform exploratory data analysis on a dataset
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        X, y = data['X'], data['y']
        feature_names = data['feature_names']
        
        print(f"\n🔍 Exploratory Data Analysis: {dataset_name.upper()}")
        print("=" * 50)
        
        # Basic statistics
        df = pd.DataFrame(X, columns=feature_names)
        if y is not None:
            df['target'] = y
            
        print("Dataset shape:", X.shape)
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        # Correlation matrix for first 10 features
        n_features = min(10, X.shape[1])
        corr_data = df.iloc[:, :n_features]
        correlation_matrix = corr_data.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix - {dataset_name} (first {n_features} features)')
        plt.tight_layout()
        plt.show()
        
        # Distribution plots for first 4 features
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(min(4, X.shape[1])):
            axes[i].hist(X[:, i], bins=20, alpha=0.7)
            axes[i].set_title(f'Distribution of {feature_names[i]}')
            axes[i].set_xlabel(feature_names[i])
            axes[i].set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.show()
        
        # Class distribution (for classification tasks)
        if y is not None and data['task'] == 'classification':
            plt.figure(figsize=(8, 6))
            unique, counts = np.unique(y, return_counts=True)
            target_names = data['target_names'] if data['target_names'] is not None else unique
            plt.bar(range(len(unique)), counts)
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.xticks(range(len(unique)), target_names, rotation=45)
            plt.tight_layout()
            plt.show()
            
            print("\nClass distribution:")
            for i, count in enumerate(counts):
                print(f"  {target_names[i]}: {count}")
        
        return df
    
    def feature_engineering(self, dataset_name='iris'):
        """
        Perform feature engineering on a dataset
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        X, y = data['X'], data['y']
        
        print(f"\n⚙️ Feature Engineering: {dataset_name.upper()}")
        print("=" * 40)
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Feature selection (for datasets with many features)
        if X.shape[1] > 5:
            selector = SelectKBest(score_func=f_classif, k=5)
            X_selected = selector.fit_transform(X_scaled, y) if y is not None else X_scaled
            selected_features = selector.get_support(indices=True)
            print(f"Selected top 5 features: {[data['feature_names'][i] for i in selected_features]}")
        else:
            X_selected = X_scaled
            print("Using all features (less than 5 features in dataset)")
            
        # Polynomial features (for demonstration)
        if X.shape[1] <= 5:  # Only for small feature sets to avoid explosion
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)
            print(f"Created polynomial features. New feature count: {X_poly.shape[1]}")
        else:
            X_poly = X_scaled
            print("Skipped polynomial features (too many initial features)")
            
        self.datasets[dataset_name]['X_processed'] = X_poly
        self.datasets[dataset_name]['scaler'] = scaler
        
        print("✅ Feature engineering completed")
        return X_poly, y
    
    def classification_models(self):
        """
        Define classification models
        """
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
    
    def regression_models(self):
        """
        Define regression models
        """
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest Regressor': RandomForestClassifier(n_estimators=100, random_state=42)
        }
    
    def train_and_evaluate_classification(self, dataset_name='iris'):
        """
        Train and evaluate classification models
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        if data['task'] != 'classification':
            print(f"Dataset {dataset_name} is not a classification dataset!")
            return
            
        X = data.get('X_processed', data['X'])
        y = data['y']
        
        print(f"\n🎯 Classification: {dataset_name.upper()}")
        print("=" * 40)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get models
        models = self.classification_models()
        results = {}
        
        print("Training models...")
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        
        # Detailed evaluation of best model
        best_result = results[best_model_name]
        print("\nClassification Report:")
        print(classification_report(best_result['y_test'], best_result['y_pred'], 
                                  target_names=data['target_names']))
        
        # Confusion Matrix
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=data['target_names'],
                    yticklabels=data['target_names'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        self.models[dataset_name] = {
            'task': 'classification',
            'best_model': best_model,
            'best_model_name': best_model_name,
            'all_results': results
        }
        
        return results
    
    def train_and_evaluate_regression(self, dataset_name='regression'):
        """
        Train and evaluate regression models
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        if data['task'] != 'regression':
            print(f"Dataset {dataset_name} is not a regression dataset!")
            return
            
        X = data.get('X_processed', data['X'])
        y = data['y']
        
        print(f"\n📈 Regression: {dataset_name.upper()}")
        print("=" * 40)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Get models
        models = self.regression_models()
        results = {}
        
        print("Training models...")
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        
        # Visualization of best model
        best_result = results[best_model_name]
        plt.figure(figsize=(10, 6))
        plt.scatter(best_result['y_test'], best_result['y_pred'], alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {best_model_name}')
        plt.show()
        
        self.models[dataset_name] = {
            'task': 'regression',
            'best_model': best_model,
            'best_model_name': best_model_name,
            'all_results': results
        }
        
        return results
    
    def clustering_analysis(self, dataset_name='clustering'):
        """
        Perform clustering analysis
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        X = data['X']
        
        print(f"\n🧩 Clustering: {dataset_name.upper()}")
        print("=" * 40)
        
        # Apply different clustering algorithms
        clustering_algorithms = {
            'K-Means': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative': AgglomerativeClustering(n_clusters=3)
        }
        
        results = {}
        
        for name, algorithm in clustering_algorithms.items():
            # Fit the algorithm
            y_pred = algorithm.fit_predict(X)
            
            # Calculate silhouette score (if more than 1 cluster)
            if len(np.unique(y_pred)) > 1:
                silhouette = silhouette_score(X, y_pred)
            else:
                silhouette = -1  # Invalid clustering
            
            results[name] = {
                'algorithm': algorithm,
                'labels': y_pred,
                'silhouette': silhouette
            }
            
            print(f"{name}: Silhouette Score = {silhouette:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            axes[idx].scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='viridis')
            axes[idx].set_title(f'{name} (Silhouette: {result["silhouette"]:.3f})')
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
        
        # Dendrogram for Agglomerative Clustering
        plt.figure(figsize=(12, 6))
        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
        plt.title('Dendrogram - Agglomerative Clustering')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        
        self.models[dataset_name] = {
            'task': 'clustering',
            'results': results
        }
        
        return results
    
    def anomaly_detection(self, dataset_name='anomaly'):
        """
        Perform anomaly detection
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        X = data['X']
        
        print(f"\n🚨 Anomaly Detection: {dataset_name.upper()}")
        print("=" * 40)
        
        # Different anomaly detection algorithms
        anomaly_detectors = {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'Local Outlier Factor': LocalOutlierFactor(contamination=0.1),
            'One-Class SVM': OneClassSVM(nu=0.1)
        }
        
        results = {}
        
        # Mark the actual anomalies (last 10 points we added)
        true_anomalies = np.zeros(len(X))
        true_anomalies[-10:] = 1  # Last 10 points are anomalies
        
        for name, detector in anomaly_detectors.items():
            if name == 'Local Outlier Factor':
                # LOF works differently
                y_pred = detector.fit_predict(X)
                # Convert to binary: -1 for anomaly, 1 for normal -> 1 for anomaly, 0 for normal
                y_pred = (y_pred == -1).astype(int)
            else:
                detector.fit(X)
                y_pred = detector.predict(X)
                # Convert to binary: -1 for anomaly, 1 for normal -> 1 for anomaly, 0 for normal
                y_pred = (y_pred == -1).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(true_anomalies, y_pred)
            precision = precision_score(true_anomalies, y_pred, zero_division=0)
            recall = recall_score(true_anomalies, y_pred, zero_division=0)
            
            results[name] = {
                'detector': detector,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            # Plot normal points in blue, anomalies in red
            normal_points = X[result['predictions'] == 0]
            anomaly_points = X[result['predictions'] == 1]
            
            axes[idx].scatter(normal_points[:, 0], normal_points[:, 1], 
                            c='blue', label='Normal', alpha=0.6)
            axes[idx].scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
                            c='red', label='Anomalies', alpha=0.8)
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
        
        self.models[dataset_name] = {
            'task': 'anomaly_detection',
            'results': results
        }
        
        return results
    
    def advanced_visualization(self, dataset_name='iris'):
        """
        Create advanced visualizations for the dataset
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        data = self.datasets[dataset_name]
        X = data.get('X_processed', data['X'])
        y = data['y']
        feature_names = data['feature_names']
        
        print(f"\n🎨 Advanced Visualization: {dataset_name.upper()}")
        print("=" * 45)
        
        # PCA
        pca = PCA(n_components=min(3, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # t-SNE
        if X.shape[0] <= 2000:  # t-SNE can be slow on large datasets
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
        else:
            X_tsne = None
            print("Skipped t-SNE (dataset too large)")
        
        # 3D Visualization if we have 3+ features
        if X_pca.shape[1] >= 3 and y is not None:
            fig = plt.figure(figsize=(15, 5))
            
            # 3D PCA
            ax1 = fig.add_subplot(131, projection='3d')
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            ax1.set_title('3D PCA')
            plt.colorbar(scatter, ax=ax1)
            
            # 2D PCA
            ax2 = fig.add_subplot(132)
            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax2.set_title('2D PCA')
            plt.colorbar(scatter, ax=ax2)
            
            # t-SNE
            if X_tsne is not None:
                ax3 = fig.add_subplot(133)
                scatter = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
                ax3.set_xlabel('t-SNE 1')
                ax3.set_ylabel('t-SNE 2')
                ax3.set_title('t-SNE')
                plt.colorbar(scatter, ax=ax3)
            
            plt.tight_layout()
            plt.show()
        else:
            # 2D visualization only
            plt.figure(figsize=(12, 5))
            
            # 2D PCA
            plt.subplot(1, 2, 1)
            if y is not None:
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
                plt.colorbar(scatter)
            else:
                plt.scatter(X_pca[:, 0], X_pca[:, 1])
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.title('2D PCA')
            
            # t-SNE or feature comparison
            plt.subplot(1, 2, 2)
            if X_tsne is not None:
                if y is not None:
                    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
                    plt.colorbar(scatter)
                else:
                    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.title('t-SNE')
            else:
                # Feature comparison if no t-SNE
                if X.shape[1] >= 2:
                    if y is not None:
                        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
                        plt.colorbar(scatter)
                    else:
                        plt.scatter(X[:, 0], X[:, 1])
                    plt.xlabel(feature_names[0])
                    plt.ylabel(feature_names[1])
                    plt.title('Feature Comparison')
            
            plt.tight_layout()
            plt.show()
        
        # Explained variance plot for PCA
        plt.figure(figsize=(10, 6))
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.show()
        
        print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
        print(f"Total Variance Explained by {pca.n_components_} PCs: {sum(pca.explained_variance_ratio_):.2%}")
    
    def model_interpretability(self, dataset_name='iris'):
        """
        Provide model interpretability features
        """
        if dataset_name not in self.models:
            print(f"No trained model found for {dataset_name}!")
            return
            
        model_info = self.models[dataset_name]
        data = self.datasets[dataset_name]
        
        print(f"\n🧠 Model Interpretability: {dataset_name.upper()}")
        print("=" * 45)
        
        if model_info['task'] == 'classification':
            best_model = model_info['best_model']
            best_model_name = model_info['best_model_name']
            
            # Feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_names = data['feature_names']
                
                # Create DataFrame for better visualization
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(importances)],  # Handle cases where we selected features
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("Feature Importances:")
                print(importance_df)
                
                # Plot feature importances
                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title(f'Feature Importances - {best_model_name}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.show()
                
            elif hasattr(best_model, 'coef_'):  # For linear models
                coef = best_model.coef_
                if coef.ndim == 1:  # Binary classification
                    coef = coef.reshape(1, -1)
                
                feature_names = data['feature_names']
                coef_df = pd.DataFrame(coef, 
                                     columns=feature_names[:coef.shape[1]],
                                     index=[f'Class {i}' for i in range(coef.shape[0])])
                
                print("Model Coefficients:")
                print(coef_df)
                
                # Plot coefficients
                plt.figure(figsize=(12, 6))
                sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
                plt.title(f'Model Coefficients - {best_model_name}')
                plt.tight_layout()
                plt.show()
            
            # Partial dependence might be too complex for this example
            print("\n✅ Model interpretability analysis complete")
    
    def generate_report(self):
        """
        Generate a comprehensive report of all analyses
        """
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ML ANALYSIS REPORT")
        print("=" * 60)
        
        print("\n📊 DATASETS ANALYZED:")
        for name, data in self.datasets.items():
            print(f"  • {name.capitalize()}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
        
        print("\n🎯 MODEL PERFORMANCE:")
        for dataset_name, model_info in self.models.items():
            if model_info['task'] == 'classification':
                best_model_name = model_info['best_model_name']
                best_result = model_info['all_results'][best_model_name]
                print(f"  • {dataset_name.capitalize()} - Best Model: {best_model_name}")
                print(f"    Accuracy: {best_result['accuracy']:.4f}, F1-Score: {best_result['f1']:.4f}")
            
            elif model_info['task'] == 'regression':
                best_model_name = model_info['best_model_name']
                best_result = model_info['all_results'][best_model_name]
                print(f"  • {dataset_name.capitalize()} - Best Model: {best_model_name}")
                print(f"    RMSE: {best_result['rmse']:.4f}, R²: {best_result['r2']:.4f}")
            
            elif model_info['task'] == 'clustering':
                results = model_info['results']
                best_algo = max(results, key=lambda x: results[x]['silhouette'])
                print(f"  • {dataset_name.capitalize()} - Best Algorithm: {best_algo}")
                print(f"    Silhouette Score: {results[best_algo]['silhouette']:.4f}")
            
            elif model_info['task'] == 'anomaly_detection':
                results = model_info['results']
                best_detector = max(results, key=lambda x: results[x]['f1'])
                print(f"  • {dataset_name.capitalize()} - Best Detector: {best_detector}")
                print(f"    Accuracy: {results[best_detector]['accuracy']:.4f}")
        
        print("\n✨ KEY FEATURES DEMONSTRATED:")
        print("  • Multiple ML tasks: Classification, Regression, Clustering, Anomaly Detection")
        print("  • Data preprocessing and feature engineering")
        print("  • Model comparison and selection")
        print("  • Cross-validation and hyperparameter tuning")
        print("  • Advanced visualization (PCA, t-SNE, 3D plots)")
        print("  • Model interpretability and feature importance")
        print("  • Comprehensive evaluation metrics")
        
        print("\n🚀 THE ULTIMATE MACHINE LEARNING SUITE!")
        print("=" * 60)

def run_comprehensive_ml_suite():
    """
    Run the comprehensive ML suite
    """
    print("🤖 WELCOME TO THE ULTIMATE MACHINE LEARNING SUITE")
    print("=" * 60)
    
    # Create ML suite instance
    ml_suite = ComprehensiveMLSuite()
    
    # Load datasets
    ml_suite.load_datasets()
    
    # Perform EDA on Iris dataset
    ml_suite.exploratory_data_analysis('iris')
    
    # Feature engineering
    ml_suite.feature_engineering('iris')
    
    # Classification
    ml_suite.train_and_evaluate_classification('iris')
    
    # Regression
    ml_suite.train_and_evaluate_regression('regression')
    
    # Clustering
    ml_suite.clustering_analysis('clustering')
    
    # Anomaly detection
    ml_suite.anomaly_detection('anomaly')
    
    # Advanced visualization
    ml_suite.advanced_visualization('iris')
    
    # Model interpretability
    ml_suite.model_interpretability('iris')
    
    # Generate final report
    ml_suite.generate_report()

# =============================================================================
# SECTION 5: AUTOMATED ML PIPELINE
# =============================================================================

class AutoMLPipeline:
    """
    Automated Machine Learning Pipeline for Classification Tasks
    """
    
    def __init__(self):
        self.datasets = {}
        self.best_pipelines = {}
        self.results = {}
        
    def load_datasets(self):
        """
        Load multiple datasets for demonstration
        """
        print("📥 Loading datasets...")
        
        datasets_info = {
            'iris': load_iris(),
            'wine': load_wine(),
            'breast_cancer': load_breast_cancer()
        }
        
        for name, data in datasets_info.items():
            self.datasets[name] = {
                'X': data.data,
                'y': data.target,
                'feature_names': data.feature_names,
                'target_names': data.target_names
            }
            
        print(f"✅ Loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def create_pipelines(self):
        """
        Create ML pipelines with different combinations of preprocessors and estimators
        """
        pipelines = {
            'rf_pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            
            'svm_pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('classifier', SVC(random_state=42, probability=True))
            ]),
            
            'lr_pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            
            'gb_pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            
            'knn_pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('classifier', KNeighborsClassifier())
            ])
        }
        
        return pipelines
    
    def get_hyperparameter_grids(self):
        """
        Define hyperparameter grids for each pipeline
        """
        param_grids = {
            'rf_pipeline': {
                'selector__k': [5, 10, 'all'],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            
            'svm_pipeline': {
                'selector__k': [5, 10, 'all'],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            },
            
            'lr_pipeline': {
                'selector__k': [5, 10, 'all'],
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            },
            
            'gb_pipeline': {
                'selector__k': [5, 10, 'all'],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            },
            
            'knn_pipeline': {
                'selector__k': [5, 10, 'all'],
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        }
        
        return param_grids
    
    def automated_model_selection(self, dataset_name='iris', search_type='grid', cv=5):
        """
        Automatically select and tune the best model for a dataset
        """
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
            
        print(f"\n🤖 AUTOMATED ML PIPELINE: {dataset_name.upper()}")
        print("=" * 50)
        
        # Get data
        data = self.datasets[dataset_name]
        X, y = data['X'], data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Dataset shape: {X.shape}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {len(np.unique(y))}")
        
        # Create pipelines and parameter grids
        pipelines = self.create_pipelines()
        param_grids = self.get_hyperparameter_grids()
        
        # Store results
        pipeline_results = {}
        
        print(f"\n🔍 Running {search_type} search for all pipelines...")
        
        # Evaluate each pipeline
        for name, pipeline in pipelines.items():
            print(f"\nProcessing {name}...")
            start_time = time.time()
            
            # Set up search
            if search_type == 'grid':
                search = GridSearchCV(
                    pipeline, 
                    param_grids[name], 
                    cv=cv, 
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    pipeline, 
                    param_grids[name], 
                    cv=cv, 
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0,
                    n_iter=20  # Number of parameter settings sampled
                )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Calculate time
            elapsed_time = time.time() - start_time
            
            # Evaluate on test set
            y_pred = search.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            pipeline_results[name] = {
                'best_pipeline': search.best_estimator_,
                'best_params': search.best_params_,
                'cv_score': search.best_score_,
                'test_accuracy': test_accuracy,
                'training_time': elapsed_time,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"  CV Score: {search.best_score_:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Training Time: {elapsed_time:.2f}s")
        
        # Find the best pipeline
        best_pipeline_name = max(pipeline_results, key=lambda x: pipeline_results[x]['test_accuracy'])
        best_pipeline = pipeline_results[best_pipeline_name]['best_pipeline']
        
        print(f"\n🏆 BEST PIPELINE: {best_pipeline_name}")
        print(f"   Test Accuracy: {pipeline_results[best_pipeline_name]['test_accuracy']:.4f}")
        print(f"   CV Score: {pipeline_results[best_pipeline_name]['cv_score']:.4f}")
        print(f"   Training Time: {pipeline_results[best_pipeline_name]['training_time']:.2f}s")
        
        # Detailed results for best pipeline
        best_result = pipeline_results[best_pipeline_name]
        print(f"\n⚙️ Best Parameters:")
        for param, value in best_result['best_params'].items():
            print(f"   {param}: {value}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(
            best_result['y_test'], 
            best_result['y_pred'], 
            target_names=data['target_names']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=data['target_names'],
                    yticklabels=data['target_names'])
        plt.title(f'Confusion Matrix - {best_pipeline_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Store results
        self.best_pipelines[dataset_name] = {
            'pipeline': best_pipeline,
            'name': best_pipeline_name,
            'results': pipeline_results
        }
        
        return pipeline_results
    
    def compare_datasets(self):
        """
        Compare the best models across different datasets
        """
        print(f"\n📊 CROSS-DATASET COMPARISON")
        print("=" * 40)
        
        comparison_data = []
        
        for dataset_name, pipeline_info in self.best_pipelines.items():
            best_result = pipeline_info['results'][pipeline_info['name']]
            comparison_data.append({
                'Dataset': dataset_name.capitalize(),
                'Best Model': pipeline_info['name'].replace('_pipeline', '').upper(),
                'Test Accuracy': best_result['test_accuracy'],
                'CV Score': best_result['cv_score'],
                'Training Time (s)': best_result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy comparison
        axes[0].bar(comparison_df['Dataset'], comparison_df['Test Accuracy'])
        axes[0].set_title('Test Accuracy Comparison')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # CV Score comparison
        axes[1].bar(comparison_df['Dataset'], comparison_df['CV Score'])
        axes[1].set_title('Cross-Validation Score Comparison')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[2].bar(comparison_df['Dataset'], comparison_df['Training Time (s)'])
        axes[2].set_title('Training Time Comparison')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance for tree-based models
        """
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        for dataset_name, pipeline_info in self.best_pipelines.items():
            pipeline = pipeline_info['pipeline']
            dataset = self.datasets[dataset_name]
            
            # Check if the pipeline has a tree-based model
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                # Get feature selector information
                selector = pipeline.named_steps['selector']
                classifier = pipeline.named_steps['classifier']
                
                # Get selected features
                if hasattr(selector, 'get_support'):
                    selected_features = selector.get_support(indices=True)
                    feature_names = [dataset['feature_names'][i] for i in selected_features]
                    importances = classifier.feature_importances_
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n{dataset_name.capitalize()} Dataset - Top 10 Features:")
                    print(importance_df.head(10))
                    
                    # Plot feature importances
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
                    plt.title(f'Top 10 Feature Importances - {dataset_name.capitalize()}')
                    plt.xlabel('Importance')
                    plt.tight_layout()
                    plt.show()
    
    def generate_automl_report(self):
        """
        Generate a comprehensive AutoML report
        """
        print("\n" + "=" * 60)
        print("🏁 AUTOMATED ML PIPELINE REPORT")
        print("=" * 60)
        
        print("\n📋 AUTOMATION SUMMARY:")
        print("✅ Automatic dataset loading and preprocessing")
        print("✅ Pipeline creation with multiple algorithms")
        print("✅ Hyperparameter optimization (Grid/Random search)")
        print("✅ Cross-validation for robust model evaluation")
        print("✅ Model selection based on performance metrics")
        print("✅ Feature selection and dimensionality reduction")
        print("✅ Comprehensive model evaluation and visualization")
        print("✅ Cross-dataset performance comparison")
        
        print("\n🎯 KEY FEATURES:")
        print("• 5 Different ML Algorithms (RF, SVM, LR, GB, KNN)")
        print("• Automated Feature Selection")
        print("• Hyperparameter Tuning")
        print("• Cross-Validation")
        print("• Performance Visualization")
        print("• Model Interpretability")
        
        print("\n🚀 AUTOML PIPELINE COMPLETE!")
        print("=" * 60)

def run_automl_pipeline():
    """
    Run the automated ML pipeline
    """
    print("🤖 WELCOME TO THE AUTOMATED ML PIPELINE")
    print("=" * 50)
    
    # Import time here to avoid issues
    import time
    
    # Create AutoML pipeline instance
    automl = AutoMLPipeline()
    
    # Load datasets
    automl.load_datasets()
    
    # Run automated model selection for each dataset
    datasets = ['iris', 'wine', 'breast_cancer']
    for dataset in datasets:
        automl.automated_model_selection(dataset, search_type='grid')
    
    # Compare datasets
    automl.compare_datasets()
    
    # Feature importance analysis
    automl.feature_importance_analysis()
    
    # Generate final report
    automl.generate_automl_report()

# =============================================================================
# SECTION 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run all ML examples
    """
    print("🤖 WELCOME TO THE ULTIMATE ML EXAMPLE SUITE")
    print("=" * 60)
    print("This script will run all machine learning examples in sequence.")
    print("Estimated time: 10-15 minutes depending on your system.")
    print("=" * 60)
    
    # Run basic ML example
    basic_ml_example()
    
    # Run advanced ML example
    run_advanced_ml_example()
    
    # Run neural network example
    neural_network_example()
    
    # Run comprehensive ML suite
    run_comprehensive_ml_suite()
    
    # Run automated ML pipeline
    run_automl_pipeline()
    
    print("\n" + "=" * 60)
    print("🎉 ALL ML EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\n📚 Summary of what you've learned:")
    print("• Basic ML classification with scikit-learn")
    print("• Advanced ML with multiple algorithms and evaluation techniques")
    print("• Neural networks with TensorFlow/Keras")
    print("• Comprehensive ML suite with clustering and anomaly detection")
    print("• Automated ML pipeline with hyperparameter tuning")
    print("\n🔧 Key techniques covered:")
    print("  - Data preprocessing and feature engineering")
    print("  - Model selection and evaluation")
    print("  - Cross-validation and hyperparameter tuning")
    print("  - Visualization and model interpretability")
    print("  - Classification, regression, clustering, and anomaly detection")
    print("  - Deep learning and automated machine learning")
    print("\n🎓 You now have a comprehensive understanding of ML workflows!")

if __name__ == "__main__":
    main()