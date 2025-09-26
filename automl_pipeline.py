import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import time

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

def main():
    """
    Main function to run the AutoML pipeline
    """
    print("🤖 WELCOME TO THE AUTOMATED ML PIPELINE")
    print("=" * 50)
    
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

if __name__ == "__main__":
    main()