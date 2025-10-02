"""
Cardiovascular Disease Prediction - Complete ML Pipeline
=========================================================
A comprehensive machine learning solution for predicting cardiovascular disease
using patient medical records.

Required Libraries:
------------------
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn joblib

Author: Data Science Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CardiovascularDiseasePredictor:
    """
    Complete ML Pipeline for Cardiovascular Disease Prediction
    """
    
    def __init__(self, data_path):
        """
        Initialize the predictor with dataset path
        
        Parameters:
        -----------
        data_path : str
            Path to the cardiovascular disease dataset (CSV file)
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.best_model = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
    
    def load_and_explore_data(self):
        """
        Load dataset and perform initial exploration
        """
        print("=" * 80)
        print("STEP 1: DATA LOADING & EXPLORATION")
        print("=" * 80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Columns: {list(self.df.columns)}")
        
        # Display basic information
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- First Few Rows ---")
        print(self.df.head())
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        # Check for missing values
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Check class distribution
        if 'cardio' in self.df.columns:
            target_col = 'cardio'
        else:
            # Find potential target column
            target_col = self.df.columns[-1]
        
        print(f"\n--- Target Variable Distribution ({target_col}) ---")
        print(self.df[target_col].value_counts())
        print(f"\nClass Balance: {self.df[target_col].value_counts(normalize=True)}")
        
        return self.df
    
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing including:
        - Outlier detection and handling
        - Feature engineering
        - Data type conversions
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 80)
        
        df = self.df.copy()
        
        # Identify target column (assuming last column or 'cardio')
        if 'cardio' in df.columns:
            target_col = 'cardio'
        else:
            target_col = df.columns[-1]
        
        print(f"\n✓ Target variable identified: '{target_col}'")
        
        # Remove ID column if exists
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            print("✓ Removed ID column")
        
        # Feature Engineering
        print("\n--- Feature Engineering ---")
        
        # 1. Age: Convert from days to years if needed
        if 'age' in df.columns and df['age'].max() > 200:
            df['age'] = (df['age'] / 365.25).round().astype(int)
            print("✓ Converted age from days to years")
        
        # 2. BMI calculation if height and weight available
        if 'height' in df.columns and 'weight' in df.columns:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            print("✓ Created BMI feature")
        
        # 3. Blood pressure features
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df['bp_diff'] = df['ap_hi'] - df['ap_lo']
            df['bp_mean'] = (df['ap_hi'] + df['ap_lo']) / 2
            print("✓ Created blood pressure difference and mean features")
        
        # Outlier Detection and Handling
        print("\n--- Outlier Detection & Handling ---")
        
        # Remove physiologically impossible values
        initial_shape = df.shape[0]
        
        # Height (should be between 120-220 cm)
        if 'height' in df.columns:
            df = df[(df['height'] >= 120) & (df['height'] <= 220)]
        
        # Weight (should be between 30-200 kg)
        if 'weight' in df.columns:
            df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
        
        # Blood pressure (systolic should be higher than diastolic)
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df = df[df['ap_hi'] > df['ap_lo']]
            df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
            df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
        
        removed = initial_shape - df.shape[0]
        print(f"✓ Removed {removed} outlier records ({removed/initial_shape*100:.2f}%)")
        print(f"  Remaining records: {df.shape[0]}")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        print(f"\n✓ Feature count: {len(self.feature_names)}")
        print(f"  Features: {self.feature_names}")
        
        return X, y
    
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """
        Split data into train/test sets and apply scaling
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        test_size : float
            Proportion of test set
        """
        print("\n" + "=" * 80)
        print("STEP 3: DATA SPLITTING & SCALING")
        print("=" * 80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        print(f"  Test size: {test_size*100}%")
        
        # Check class balance in splits
        print(f"\n  Train set class distribution:")
        print(f"    {self.y_train.value_counts(normalize=True)}")
        print(f"\n  Test set class distribution:")
        print(f"    {self.y_test.value_counts(normalize=True)}")
        
        # Scale features using RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Feature scaling applied (RobustScaler)")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    
    def handle_class_imbalance(self):
        """
        Handle class imbalance using SMOTE
        """
        print("\n" + "=" * 80)
        print("STEP 4: HANDLING CLASS IMBALANCE")
        print("=" * 80)
        
        # Check if imbalance exists
        class_dist = self.y_train.value_counts(normalize=True)
        print(f"\nOriginal class distribution:")
        print(class_dist)
        
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 1.5:
            print("\n⚠ Significant class imbalance detected. Applying SMOTE...")
            
            # Apply SMOTE
            smote = SMOTE(random_state=RANDOM_STATE)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
            
            print(f"\n✓ SMOTE applied:")
            print(f"  Before: {self.X_train_scaled.shape[0]} samples")
            print(f"  After: {self.X_train_resampled.shape[0]} samples")
            print(f"\nNew class distribution:")
            print(pd.Series(self.y_train_resampled).value_counts(normalize=True))
        else:
            print("\n✓ Class balance is acceptable. No resampling needed.")
            self.X_train_resampled = self.X_train_scaled
            self.y_train_resampled = self.y_train
    
    
    def train_models(self):
        """
        Train multiple ML models with optimized hyperparameters
        """
        print("\n" + "=" * 80)
        print("STEP 5: MODEL TRAINING")
        print("=" * 80)
        
        # Define models with tuned hyperparameters
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_STATE,
                verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                random_state=RANDOM_STATE,
                verbose=0
            )
        }
        
        # Train each model
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"\n  → Training {name}...", end=" ")
            model.fit(self.X_train_resampled, self.y_train_resampled)
            print("✓ Done")
        
        print("\n✓ All models trained successfully!")
    
    
    def evaluate_models(self):
        """
        Evaluate all trained models using multiple metrics
        """
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        self.results = {}
        
        print("\n" + "-" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 80)
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            # Print results
            print(f"{name:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")
            
            # Track best model (using F1-score as primary metric)
            if f1 > best_score:
                best_score = f1
                best_model_name = name
        
        print("-" * 80)
        print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_score:.4f})")
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        # Detailed evaluation of best model
        self._print_detailed_evaluation(best_model_name)
        
        return self.results
    
    
    def _print_detailed_evaluation(self, model_name):
        """
        Print detailed evaluation for the best model
        """
        print("\n" + "=" * 80)
        print(f"DETAILED EVALUATION: {model_name}")
        print("=" * 80)
        
        result = self.results[model_name]
        
        # Classification Report
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, result['y_pred'], 
                                   target_names=['No Disease', 'Disease']))
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = result['confusion_matrix']
        print(f"\n                Predicted")
        print(f"              No Disease  Disease")
        print(f"Actual No      {cm[0][0]:<8}    {cm[0][1]:<8}")
        print(f"       Disease {cm[1][0]:<8}    {cm[1][1]:<8}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"\n--- Additional Metrics ---")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Negative Predictive Value: {npv:.4f}")
        print(f"  True Positives: {tp}")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
    
    
    def feature_importance_analysis(self):
        """
        Analyze and display feature importance for tree-based models
        """
        if hasattr(self.best_model, 'feature_importances_'):
            print("\n" + "=" * 80)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("=" * 80)
            
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n{'Rank':<6} {'Feature':<20} {'Importance':<12}")
            print("-" * 40)
            
            for i, idx in enumerate(indices[:10], 1):
                print(f"{i:<6} {self.feature_names[idx]:<20} {importances[idx]:<12.4f}")
    
    
    def save_model(self, filepath='cardio_disease_model.pkl'):
        """
        Save the trained model and preprocessing objects
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        print("\n" + "=" * 80)
        print("STEP 7: MODEL PERSISTENCE")
        print("=" * 80)
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'results': self.results[self.best_model_name]
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n✓ Model saved successfully to: {filepath}")
        print(f"  Model: {self.best_model_name}")
        print(f"  Performance: F1-Score = {self.results[self.best_model_name]['f1_score']:.4f}")
    
    
    def predict_new_patient(self, patient_data):
        """
        Predict cardiovascular disease risk for new patient data
        
        Parameters:
        -----------
        patient_data : dict or DataFrame
            Patient features
            
        Returns:
        --------
        dict : Prediction results with probability
        """
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                if feature == 'bmi' and 'height' in patient_df.columns and 'weight' in patient_df.columns:
                    patient_df['bmi'] = patient_df['weight'] / ((patient_df['height'] / 100) ** 2)
                elif feature == 'bp_diff' and 'ap_hi' in patient_df.columns and 'ap_lo' in patient_df.columns:
                    patient_df['bp_diff'] = patient_df['ap_hi'] - patient_df['ap_lo']
                elif feature == 'bp_mean' and 'ap_hi' in patient_df.columns and 'ap_lo' in patient_df.columns:
                    patient_df['bp_mean'] = (patient_df['ap_hi'] + patient_df['ap_lo']) / 2
                else:
                    raise ValueError(f"Missing required feature: {feature}")
        
        # Select and order features
        patient_df = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.best_model.predict(patient_scaled)[0]
        probability = self.best_model.predict_proba(patient_scaled)[0]
        
        result = {
            'prediction': 'Cardiovascular Disease' if prediction == 1 else 'No Disease',
            'risk_probability': probability[1],
            'confidence': max(probability),
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    
    def _get_risk_level(self, probability):
        """
        Categorize risk level based on probability
        """
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.6:
            return 'Moderate Risk'
        else:
            return 'High Risk'


def load_saved_model(filepath='cardio_disease_model.pkl'):
    """
    Load a previously saved model
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    dict : Model package containing model, scaler, and metadata
    """
    model_package = joblib.load(filepath)
    print(f"✓ Model loaded: {model_package['model_name']}")
    print(f"  F1-Score: {model_package['results']['f1_score']:.4f}")
    return model_package


def predict_with_saved_model(model_package, patient_data):
    """
    Make predictions using a loaded model
    
    Parameters:
    -----------
    model_package : dict
        Loaded model package
    patient_data : dict or DataFrame
        Patient features
        
    Returns:
    --------
    dict : Prediction results
    """
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = patient_data.copy()
    
    # Feature engineering
    if 'bmi' not in patient_df.columns and 'height' in patient_df.columns and 'weight' in patient_df.columns:
        patient_df['bmi'] = patient_df['weight'] / ((patient_df['height'] / 100) ** 2)
    
    if 'bp_diff' not in patient_df.columns and 'ap_hi' in patient_df.columns and 'ap_lo' in patient_df.columns:
        patient_df['bp_diff'] = patient_df['ap_hi'] - patient_df['ap_lo']
        patient_df['bp_mean'] = (patient_df['ap_hi'] + patient_df['ap_lo']) / 2
    
    # Select features
    patient_df = patient_df[feature_names]
    
    # Scale and predict
    patient_scaled = scaler.transform(patient_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    return {
        'prediction': 'Cardiovascular Disease' if prediction == 1 else 'No Disease',
        'risk_probability': probability[1],
        'confidence': max(probability),
        'risk_level': 'Low Risk' if probability[1] < 0.3 else ('Moderate Risk' if probability[1] < 0.6 else 'High Risk')
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("CARDIOVASCULAR DISEASE PREDICTION - ML PIPELINE")
    print("=" * 80)
    
    # Initialize predictor
    predictor = CardiovascularDiseasePredictor('cardio_train.csv')
    
    # Execute pipeline
    try:
        # 1. Load and explore data
        predictor.load_and_explore_data()
        
        # 2. Preprocess data
        X, y = predictor.preprocess_data()
        
        # 3. Split and scale
        predictor.split_and_scale_data(X, y)
        
        # 4. Handle class imbalance
        predictor.handle_class_imbalance()
        
        # 5. Train models
        predictor.train_models()
        
        # 6. Evaluate models
        predictor.evaluate_models()
        
        # 7. Feature importance
        predictor.feature_importance_analysis()
        
        # 8. Save model
        predictor.save_model('cardio_disease_model.pkl')
        
        # 9. Demo prediction
        print("\n" + "=" * 80)
        print("DEMO: PREDICTING NEW PATIENT")
        print("=" * 80)
        
        sample_patient = {
            'age': 55,
            'gender': 2,  # Female
            'height': 165,
            'weight': 75,
            'ap_hi': 140,
            'ap_lo': 90,
            'cholesterol': 2,
            'gluc': 1,
            'smoke': 0,
            'alco': 0,
            'active': 1
        }
        
        print("\nPatient Data:")
        for key, value in sample_patient.items():
            print(f"  {key}: {value}")
        
        prediction = predictor.predict_new_patient(sample_patient)
        
        print("\n--- Prediction Results ---")
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Risk Probability: {prediction['risk_probability']:.2%}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  Risk Level: {prediction['risk_level']}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
