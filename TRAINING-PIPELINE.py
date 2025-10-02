"""
Cardiovascular Disease Prediction - TRAINING PIPELINE
======================================================
Complete training, testing, validation, and metrics calculation pipeline.
This file trains multiple models, evaluates them, and saves the best model.

Required Libraries:
------------------
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn joblib

Usage:
------
python train_model.py

Output:
-------
- cardio_disease_model.pkl (saved best model)
- Comprehensive metrics and evaluation results
- Feature importance analysis

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
                             classification_report, roc_curve, auc)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CardiovascularTrainingPipeline:
    """
    Complete Training Pipeline for Cardiovascular Disease Prediction
    """
    
    def __init__(self, data_path='cardio_train.csv'):
        """
        Initialize the training pipeline
        
        Parameters:
        -----------
        data_path : str
            Path to the cardiovascular disease dataset (CSV file)
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.scaler = None
        self.best_model = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
    
    def load_and_explore_data(self):
        """
        Load dataset and perform comprehensive exploration
        """
        print("=" * 90)
        print("STEP 1: DATA LOADING & EXPLORATION")
        print("=" * 90)
        
        # Load data
        self.df = pd.read_csv(self.data_path, delimiter=';')
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Display basic information
        print("\n--- Dataset Structure ---")
        print(f"  Columns: {list(self.df.columns)}")
        print(f"\n  Data Types:")
        for col, dtype in self.df.dtypes.items():
            print(f"    {col:<15} : {dtype}")
        
        print("\n--- Statistical Summary ---")
        print(self.df.describe())
        
        # Check for missing values
        print("\n--- Missing Values Check ---")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values found!")
        else:
            print("  Missing values per column:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count} ({count/len(self.df)*100:.2f}%)")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n--- Duplicate Rows ---")
        print(f"  Duplicate rows: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        # Target variable analysis
        if 'cardio' in self.df.columns:
            target_col = 'cardio'
        else:
            target_col = self.df.columns[-1]
        
        print(f"\n--- Target Variable: '{target_col}' ---")
        target_counts = self.df[target_col].value_counts().sort_index()
        print(f"  Distribution:")
        for val, count in target_counts.items():
            label = "No Disease" if val == 0 else "Disease"
            print(f"    {label} ({val}): {count:,} ({count/len(self.df)*100:.2f}%)")
        
        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"\n  Class imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 1.5:
            print("  ⚠ Significant class imbalance detected - will apply SMOTE")
        else:
            print("  ✓ Classes are relatively balanced")
        
        # Feature value ranges
        print("\n--- Feature Value Ranges ---")
        for col in self.df.columns:
            if col != target_col and col != 'id':
                print(f"  {col:<15} : min={self.df[col].min():.2f}, max={self.df[col].max():.2f}, mean={self.df[col].mean():.2f}")
        
        return self.df
    
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing with feature engineering
        """
        print("\n" + "=" * 90)
        print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 90)
        
        df = self.df.copy()
        initial_rows = len(df)
        
        # Identify target column
        if 'cardio' in df.columns:
            target_col = 'cardio'
        else:
            target_col = df.columns[-1]
        
        print(f"\n✓ Target variable: '{target_col}'")
        
        # Remove ID column if exists
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            print("✓ Removed ID column")
        
        # Feature Engineering
        print("\n--- Feature Engineering ---")
        
        # 1. Age: Convert from days to years
        if 'age' in df.columns and df['age'].max() > 200:
            df['age'] = (df['age'] / 365.25).round().astype(int)
            print(f"  ✓ Converted age from days to years (range: {df['age'].min()}-{df['age'].max()} years)")
        
        # 2. BMI calculation
        if 'height' in df.columns and 'weight' in df.columns:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            print(f"  ✓ Created BMI feature (range: {df['bmi'].min():.2f}-{df['bmi'].max():.2f})")
        
        # 3. Blood pressure features
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df['bp_diff'] = df['ap_hi'] - df['ap_lo']  # Pulse pressure
            df['bp_mean'] = (df['ap_hi'] + df['ap_lo']) / 2  # Mean arterial pressure approximation
            print(f"  ✓ Created pulse pressure (bp_diff) and mean BP (bp_mean)")
        
        # 4. Age groups (categorical from numerical)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 40, 50, 60, 100], 
                                     labels=[0, 1, 2, 3])
            df['age_group'] = df['age_group'].astype(int)
            print(f"  ✓ Created age_group feature (0:<40, 1:40-50, 2:50-60, 3:60+)")
        
        # Outlier Detection and Removal
        print("\n--- Outlier Detection & Removal ---")
        print(f"  Initial dataset size: {initial_rows:,} rows")
        
        # Remove physiologically impossible values
        outlier_conditions = []
        
        # Height (120-220 cm is physiologically reasonable)
        if 'height' in df.columns:
            before = len(df)
            df = df[(df['height'] >= 120) & (df['height'] <= 220)]
            removed = before - len(df)
            if removed > 0:
                print(f"    - Height outliers removed: {removed} rows")
        
        # Weight (30-200 kg is physiologically reasonable)
        if 'weight' in df.columns:
            before = len(df)
            df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
            removed = before - len(df)
            if removed > 0:
                print(f"    - Weight outliers removed: {removed} rows")
        
        # Blood pressure (systolic > diastolic, reasonable ranges)
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            before = len(df)
            df = df[df['ap_hi'] > df['ap_lo']]  # Systolic must be higher
            df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
            df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 180)]
            removed = before - len(df)
            if removed > 0:
                print(f"    - Blood pressure outliers removed: {removed} rows")
        
        # BMI (12-60 is reasonable range)
        if 'bmi' in df.columns:
            before = len(zgodovef)
            df = df[(df['bmi'] >= 12) & (df['bmi'] <= 60)]
            removed = before - len(df)
            if removed > 0:
                print(f"    - BMI outliers removed: {removed} rows")
        
        total_removed = initial_rows - len(df)
        print(f"\n  ✓ Total outliers removed: {total_removed:,} rows ({total_removed/initial_rows*100:.2f}%)")
        print(f"  ✓ Final dataset size: {len(df):,} rows")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        print(f"\n✓ Final feature count: {len(self.feature_names)}")
        print(f"  Features: {self.feature_names}")
        
        return X, y
    
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """
        Split data into train, validation, and test sets
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        test_size : float
            Proportion of test set
        val_size : float
            Proportion of validation set
        """
        print("\n" + "=" * 90)
        print("STEP 3: DATA SPLITTING (TRAIN / VALIDATION / TEST)")
        print("=" * 90)
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  Training set:   {self.X_train.shape[0]:,} samples ({self.X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Validation set: {self.X_val.shape[0]:,} samples ({self.X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set:       {self.X_test.shape[0]:,} samples ({self.X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Check class distribution in each split
        print(f"\n--- Class Distribution ---")
        print(f"  Training set:")
        train_dist = self.y_train.value_counts(normalize=True).sort_index()
        for val, pct in train_dist.items():
            print(f"    Class {val}: {pct*100:.2f}%")
        
        print(f"\n  Validation set:")
        val_dist = self.y_val.value_counts(normalize=True).sort_index()
        for val, pct in val_dist.items():
            print(f"    Class {val}: {pct*100:.2f}%")
        
        print(f"\n  Test set:")
        test_dist = self.y_test.value_counts(normalize=True).sort_index()
        for val, pct in test_dist.items():
            print(f"    Class {val}: {pct*100:.2f}%")
    
    
    def scale_features(self):
        """
        Scale features using RobustScaler
        """
        print("\n" + "=" * 90)
        print("STEP 4: FEATURE SCALING")
        print("=" * 90)
        
        # Use RobustScaler (less sensitive to outliers than StandardScaler)
        self.scaler = RobustScaler()
        
        # Fit on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Feature scaling applied using RobustScaler")
        print("  (RobustScaler is more robust to outliers than StandardScaler)")
        print(f"\n  Scaled feature statistics (training set):")
        scaled_df = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        print(f"    Mean range: [{scaled_df.mean().min():.3f}, {scaled_df.mean().max():.3f}]")
        print(f"    Std range:  [{scaled_df.std().min():.3f}, {scaled_df.std().max():.3f}]")
    
    
    def handle_class_imbalance(self):
        """
        Handle class imbalance using SMOTE on training data only
        """
        print("\n" + "=" * 90)
        print("STEP 5: HANDLING CLASS IMBALANCE (SMOTE)")
        print("=" * 90)
        
        # Check imbalance
        class_dist = self.y_train.value_counts(normalize=True)
        print(f"\nOriginal training class distribution:")
        for val, pct in class_dist.sort_index().items():
            print(f"  Class {val}: {pct*100:.2f}%")
        
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.2:
            print("\n⚠ Class imbalance detected. Applying SMOTE...")
            
            # Apply SMOTE to training data only
            smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )
            
            print(f"\n✓ SMOTE applied successfully:")
            print(f"  Before: {self.X_train_scaled.shape[0]:,} samples")
            print(f"  After:  {self.X_train_resampled.shape[0]:,} samples")
            print(f"\nNew training class distribution:")
            new_dist = pd.Series(self.y_train_resampled).value_counts(normalize=True)
            for val, pct in new_dist.sort_index().items():
                print(f"  Class {val}: {pct*100:.2f}%")
        else:
            print("\n✓ Classes are relatively balanced. No resampling needed.")
            self.X_train_resampled = self.X_train_scaled
            self.y_train_resampled = self.y_train
    
    
    def train_models(self):
        """
        Train multiple ML models with optimized hyperparameters
        """
        print("\n" + "=" * 90)
        print("STEP 6: MODEL TRAINING")
        print("=" * 90)
        
        # Define models with tuned hyperparameters
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                C=0.1
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
                min_samples_split=5,
                random_state=RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
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
        
        print(f"\n✓ Training {len(self.models)} models on {self.X_train_resampled.shape[0]:,} samples...")
        print(f"  Features: {self.X_train_resampled.shape[1]}")
        print()
        
        # Train each model
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"  [{i}/{len(self.models)}] Training {name}...", end=" ")
            model.fit(self.X_train_resampled, self.y_train_resampled)
            print("✓")
        
        print(f"\n✓ All models trained successfully!")
    
    
    def validate_models(self):
        """
        Validate all models on the validation set
        """
        print("\n" + "=" * 90)
        print("STEP 7: MODEL VALIDATION")
        print("=" * 90)
        
        print(f"\nValidating models on validation set ({self.X_val_scaled.shape[0]:,} samples)...\n")
        
        self.val_results = {}
        
        print("-" * 90)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 90)
        
        for name, model in self.models.items():
            # Predictions on validation set
            y_pred = model.predict(self.X_val_scaled)
            y_pred_proba = model.predict_proba(self.X_val_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred)
            recall = recall_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            roc_auc = roc_auc_score(self.y_val, y_pred_proba)
            
            self.val_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(self.y_val, y_pred)
            }
            
            print(f"{name:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")
        
        print("-" * 90)
    
    
    def test_models(self):
        """
        Test all models on the held-out test set and select best model
        """
        print("\n" + "=" * 90)
        print("STEP 8: MODEL TESTING (FINAL EVALUATION)")
        print("=" * 90)
        
        print(f"\nTesting models on test set ({self.X_test_scaled.shape[0]:,} samples)...\n")
        
        self.test_results = {}
        
        print("-" * 90)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 90)
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            # Predictions on test set
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            cm = confusion_matrix(self.y_test, y_pred)
            
            self.test_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': cm
            }
            
            print(f"{name:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {roc_auc:<12.4f}")
            
            # Track best model using weighted score (F1 + ROC-AUC)
            combined_score = (f1 * 0.6) + (roc_auc * 0.4)
            if combined_score > best_score:
                best_score = combined_score
                best_model_name = name
        
        print("-" * 90)
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   Combined Score: {best_score:.4f} (0.6×F1 + 0.4×ROC-AUC)")
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return best_model_name
    
    
    def detailed_model_analysis(self):
        """
        Comprehensive analysis of the best model
        """
        print("\n" + "=" * 90)
        print(f"DETAILED ANALYSIS: {self.best_model_name}")
        print("=" * 90)
        
        result = self.test_results[self.best_model_name]
        
        # Classification Report
        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, result['y_pred'], 
                                   target_names=['No Disease (0)', 'Disease (1)'],
                                   digits=4))
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = result['confusion_matrix']
        print(f"\n                    Predicted")
        print(f"                 No Disease    Disease")
        print(f"Actual No Disease   {cm[0][0]:<8}      {cm[0][1]:<8}")
        print(f"       Disease      {cm[1][0]:<8}      {cm[1][1]:<8}")
        
        # Additional Metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n--- Extended Metrics ---")
        print(f"  True Positives (TP):  {tp:,}")
        print(f"  True Negatives (TN):  {tn:,}")
        print(f"  False Positives (FP): {fp:,}")
        print(f"  False Negatives (FN): {fn:,}")
        print(f"\n  Sensitivity (Recall):         {sensitivity:.4f}")
        print(f"  Specificity:                  {specificity:.4f}")
        print(f"  Positive Predictive Value:    {ppv:.4f}")
        print(f"  Negative Predictive Value:    {npv:.4f}")
        
        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n--- Feature Importance (Top 10) ---")
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n{'Rank':<6} {'Feature':<20} {'Importance':<12} {'Cumulative':<12}")
            print("-" * 52)
            
            cumsum = 0
            for i, idx in enumerate(indices[:10], 1):
                cumsum += importances[idx]
                print(f"{i:<6} {self.feature_names[idx]:<20} {importances[idx]:<12.4f} {cumsum:<12.4f}")
        
        # Cross-validation score
        print(f"\n--- Cross-Validation (5-Fold) ---")
        cv_scores = cross_val_score(self.best_model, self.X_train_resampled, 
                                    self.y_train_resampled, cv=5, scoring='f1')
        print(f"  CV F1-Scores: {cv_scores}")
        print(f"  Mean CV F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    
    def save_model(self, filepath='cardio_disease_model.pkl'):
        """
        Save the best model and all necessary components
        """
        print("\n" + "=" * 90)
        print("STEP 9: MODEL PERSISTENCE")
        print("=" * 90)
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'test_results': self.test_results[self.best_model_name],
            'val_results': self.val_results[self.best_model_name],
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_importance': self.best_model.feature_importances_ if hasattr(self.best_model, 'feature_importances_') else None
        }
        
        joblib.dump(model_package, filepath)
        
        print(f"\n✓ Model saved successfully!")
        print(f"  Filepath: {filepath}")
        print(f"  Model: {self.best_model_name}")
        print(f"  Test F1-Score: {self.test_results[self.best_model_name]['f1_score']:.4f}")
        print(f"  Test ROC-AUC: {self.test_results[self.best_model_name]['roc_auc']:.4f}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "=" * 90)
        print("TRAINING SUMMARY REPORT")
        print("=" * 90)
        
        print(f"\n📊 Dataset Information:")
        print(f"  Total samples processed: {len(self.y_train) + len(self.y_val) + len(self.y_test):,}")
        print(f"  Training samples: {len(self.y_train):,}")
        print(f"  Validation samples: {len(self.y_val):,}")
        print(f"  Test samples: {len(self.y_test):,}")
        print(f"  Number of features: {len(self.feature_names)}")
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        
        print(f"\n📈 Performance Metrics:")
        best_result = self.test_results[self.best_model_name]
        print(f"  Accuracy:  {best_result['accuracy']:.4f}")
        print(f"  Precision: {best_result['precision']:.4f}")
        print(f"  Recall:    {best_result['recall']:.4f}")
        print(f"  F1-Score:  {best_result['f1_score']:.4f}")
        print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")
        
        print(f"\n✅ Model ready for deployment!")
        print(f"   Use 'predict_model.py' to make predictions on new patients")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    import os
    
    print("\n" + "=" * 90)
    print(" CARDIOVASCULAR DISEASE PREDICTION - TRAINING PIPELINE")
    print("=" * 90)
    print("\n This pipeline will:")
    print("  1. Load and explore the dataset")
    print("  2. Preprocess data and engineer features")
    print("  3. Split data into train/validation/test sets")
    print("  4. Scale features")
    print("  5. Handle class imbalance with SMOTE")
    print("  6. Train 6 different ML models")
    print("  7. Validate models")
    print("  8. Test models and select the best one")
    print("  9. Save the best model for future use")
    print("\n" + "=" * 90)
    
    # Initialize pipeline
    pipeline = CardiovascularTrainingPipeline('cardio_train.csv')
    
    try:
        # Execute complete training pipeline
        pipeline.load_and_explore_data()
        X, y = pipeline.preprocess_data()
        pipeline.split_data(X, y)
        pipeline.scale_features()
        pipeline.handle_class_imbalance()
        pipeline.train_models()
        pipeline.validate_models()
        pipeline.test_models()
        pipeline.detailed_model_analysis()
        pipeline.save_model('cardio_disease_model.pkl')
        pipeline.generate_summary_report()
        
        print("\n" + "=" * 90)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        print("\nNext steps:")
        print("  1. Review the metrics above")
        print("  2. Use 'predict_model.py' to make predictions")
        print("  3. Model saved as: cardio_disease_model.pkl")
        print("\n")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Could not find 'cardio_train.csv'")
        print("   Please ensure the dataset file is in the same directory as this script.")
        print("   Download from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and fix any issues.")
