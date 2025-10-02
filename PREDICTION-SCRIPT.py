"""
Cardiovascular Disease Prediction - PREDICTION SCRIPT
======================================================
Load trained model and make predictions on new patient data.

Required Libraries:
------------------
pip install pandas numpy joblib

Usage:
------
python predict_model.py

Features Required for Prediction:
---------------------------------
- age: Age in years (or days, will be converted)
- gender: 1 = Male, 2 = Female
- height: Height in cm
- weight: Weight in kg
- ap_hi: Systolic blood pressure
- ap_lo: Diastolic blood pressure
- cholesterol: 1 = Normal, 2 = Above normal, 3 = Well above normal
- gluc: Glucose level (1 = Normal, 2 = Above normal, 3 = Well above normal)
- smoke: 0 = No, 1 = Yes
- alco: Alcohol intake (0 = No, 1 = Yes)
- active: Physical activity (0 = No, 1 = Yes)

Author: Data Science Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


class CardiovascularPredictor:
    """
    Cardiovascular Disease Prediction System
    """
    
    def __init__(self, model_path='cardio_disease_model.pkl'):
        """
        Initialize predictor by loading the trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        """
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    
    def load_model(self):
        """
        Load the trained model and preprocessing objects
        """
        try:
            print("=" * 80)
            print("LOADING TRAINED MODEL")
            print("=" * 80)
            
            self.model_package = joblib.load(self.model_path)
            self.model = self.model_package['model']
            self.scaler = self.model_package['scaler']
            self.feature_names = self.model_package['feature_names']
            
            print(f"\n✓ Model loaded successfully!")
            print(f"  Model type: {self.model_package['model_name']}")
            print(f"  Training date: {self.model_package.get('training_date', 'N/A')}")
            print(f"  Features required: {len(self.feature_names)}")
            
            # Display model performance
            if 'test_results' in self.model_package:
                results = self.model_package['test_results']
                print(f"\n  Model Performance (Test Set):")
                print(f"    Accuracy:  {results['accuracy']:.4f}")
                print(f"    Precision: {results['precision']:.4f}")
                print(f"    Recall:    {results['recall']:.4f}")
                print(f"    F1-Score:  {results['f1_score']:.4f}")
                print(f"    ROC-AUC:   {results['roc_auc']:.4f}")
            
        except FileNotFoundError:
            print(f"\n❌ ERROR: Model file not found at '{self.model_path}'")
            print("   Please run 'train_model.py' first to train and save the model.")
            raise
        except Exception as e:
            print(f"\n❌ ERROR loading model: {str(e)}")
            raise
    
    
    def preprocess_input(self, patient_data):
        """
        Preprocess patient data to match training format
        
        Parameters:
        -----------
        patient_data : dict or DataFrame
            Patient features
            
        Returns:
        --------
        DataFrame : Preprocessed features ready for prediction
        """
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Age conversion (if in days, convert to years)
        if 'age' in df.columns and df['age'].iloc[0] > 200:
            df['age'] = (df['age'] / 365.25).round().astype(int)
        
        # Feature Engineering (must match training pipeline)
        
        # 1. BMI
        if 'bmi' not in df.columns:
            if 'height' in df.columns and 'weight' in df.columns:
                df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            else:
                raise ValueError("Missing 'height' or 'weight' for BMI calculation")
        
        # 2. Blood pressure features
        if 'bp_diff' not in df.columns:
            if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
                df['bp_diff'] = df['ap_hi'] - df['ap_lo']
                df['bp_mean'] = (df['ap_hi'] + df['ap_lo']) / 2
            else:
                raise ValueError("Missing 'ap_hi' or 'ap_lo' for blood pressure features")
        
        # 3. Age group
        if 'age_group' not in df.columns:
            if 'age' in df.columns:
                df['age_group'] = pd.cut(df['age'], 
                                         bins=[0, 40, 50, 60, 100], 
                                         labels=[0, 1, 2, 3])
                df['age_group'] = df['age_group'].astype(int)
            else:
                raise ValueError("Missing 'age' for age_group calculation")
        
        # Select only required features in correct order
        try:
            df_features = df[self.feature_names]
        except KeyError as e:
            missing_features = set(self.feature_names) - set(df.columns)
            raise ValueError(f"Missing required features: {missing_features}")
        
        return df_features
    
    
    def predict(self, patient_data, verbose=True):
        """
        Make prediction for patient(s)
        
        Parameters:
        -----------
        patient_data : dict or DataFrame
            Patient features
        verbose : bool
            Whether to print detailed output
            
        Returns:
        --------
        dict or list : Prediction results
        """
        # Preprocess input
        df_processed = self.preprocess_input(patient_data)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Format results
        results = []
        for i in range(len(df_processed)):
            pred = predictions[i]
            prob = probabilities[i]
            
            result = {
                'prediction': 'Cardiovascular Disease' if pred == 1 else 'No Disease',
                'prediction_code': int(pred),
                'disease_probability': float(prob[1]),
                'no_disease_probability': float(prob[0]),
                'confidence': float(max(prob)),
                'risk_level': self._get_risk_level(prob[1])
            }
            results.append(result)
        
        # Print results if verbose
        if verbose:
            self._print_prediction_results(results, patient_data if isinstance(patient_data, dict) else None)
        
        # Return single dict if single prediction, else list
        return results[0] if len(results) == 1 else results
    
    
    def _get_risk_level(self, disease_probability):
        """
        Categorize risk level based on disease probability
        """
        if disease_probability < 0.3:
            return 'Low Risk'
        elif disease_probability < 0.6:
            return 'Moderate Risk'
        elif disease_probability < 0.8:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    
    def _print_prediction_results(self, results, patient_data=None):
        """
        Print formatted prediction results
        """
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"\n--- Patient {i+1} ---")
            
            # Print input data if available
            if patient_data is not None and len(results) == 1:
                print("\nPatient Information:")
                for key, value in patient_data.items():
                    if key not in ['bmi', 'bp_diff', 'bp_mean', 'age_group']:
                        print(f"  {key:<15}: {value}")
            
            # Print prediction
            print(f"\n🔍 Diagnosis: {result['prediction']}")
            print(f"   Risk Level: {result['risk_level']}")
            
            # Print probabilities
            print(f"\n📊 Probability Breakdown:")
            print(f"   No Disease:  {result['no_disease_probability']*100:.2f}%")
            print(f"   Disease:     {result['disease_probability']*100:.2f}%")
            print(f"   Confidence:  {result['confidence']*100:.2f}%")
            
            # Risk interpretation
            print(f"\n💡 Interpretation:")
            if result['prediction_code'] == 0:
                if result['disease_probability'] < 0.2:
                    print("   ✓ Very low risk of cardiovascular disease")
                    print("   ✓ Continue maintaining healthy lifestyle")
                else:
                    print("   ⚠ Low risk, but some risk factors may be present")
                    print("   ⚠ Consider preventive measures and regular checkups")
            else:
                if result['disease_probability'] > 0.8:
                    print("   ⚠⚠ HIGH RISK: Strong indicators of cardiovascular disease")
                    print("   ⚠⚠ Immediate medical consultation recommended")
                elif result['disease_probability'] > 0.6:
                    print("   ⚠ MODERATE-HIGH RISK: Significant risk factors present")
                    print("   ⚠ Medical evaluation strongly recommended")
                else:
                    print("   ⚠ MODERATE RISK: Some concerning factors detected")
                    print("   ⚠ Consult healthcare provider for assessment")
            
            print("\n" + "-" * 80)
    
    
    def predict_batch(self, csv_path, output_path=None):
        """
        Make predictions for multiple patients from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with patient data
        output_path : str, optional
            Path to save predictions CSV
            
        Returns:
        --------
        DataFrame : Predictions for all patients
        """
        print("\n" + "=" * 80)
        print("BATCH PREDICTION")
        print("=" * 80)
        
        # Load data
        print(f"\nLoading patient data from: {csv_path}")
        df = pd.read_csv(csv_path, delimiter=';')
        print(f"✓ Loaded {len(df)} patients")
        
        # Make predictions
        print("\nMaking predictions...")
        results = self.predict(df, verbose=False)
        
        # Add predictions to dataframe
        df_results = df.copy()
        df_results['prediction'] = [r['prediction'] for r in results]
        df_results['disease_probability'] = [r['disease_probability'] for r in results]
        df_results['risk_level'] = [r['risk_level'] for r in results]
        
        # Print summary
        print("\n" + "-" * 80)
        print("BATCH PREDICTION SUMMARY")
        print("-" * 80)
        print(f"\nTotal patients: {len(df_results)}")
        print(f"\nPrediction Distribution:")
        pred_counts = df_results['prediction'].value_counts()
        for pred, count in pred_counts.items():
            print(f"  {pred}: {count} ({count/len(df_results)*100:.1f}%)")
        
        print(f"\nRisk Level Distribution:")
        risk_counts = df_results['risk_level'].value_counts()
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count} ({count/len(df_results)*100:.1f}%)")
        
        # Save results if output path provided
        if output_path:
            df_results.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {output_path}")
        
        return df_results


# ============================================================================
# EXAMPLE USAGE FUNCTIONS
# ============================================================================

def example_single_prediction():
    """
    Example: Predict for a single patient
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SINGLE PATIENT PREDICTION")
    print("=" * 80)
    
    # Initialize predictor
    predictor = CardiovascularPredictor('cardio_disease_model.pkl')
    
    # Example patient data
    patient = {
        'age': 55,              # Age in years
        'gender': 2,            # 1=Male, 2=Female
        'height': 165,          # Height in cm
        'weight': 75,           # Weight in kg
        'ap_hi': 140,           # Systolic BP
        'ap_lo': 90,            # Diastolic BP
        'cholesterol': 2,       # 1=Normal, 2=Above, 3=Well above
        'gluc': 1,              # Glucose: 1=Normal, 2=Above, 3=Well above
        'smoke': 0,             # 0=No, 1=Yes
        'alco': 0,              # 0=No, 1=Yes
        'active': 1             # 0=No, 1=Yes
    }
    
    # Make prediction
    result = predictor.predict(patient)
    
    return result


def example_multiple_patients():
    """
    Example: Predict for multiple patients
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MULTIPLE PATIENTS PREDICTION")
    print("=" * 80)
    
    # Initialize predictor
    predictor = CardiovascularPredictor('cardio_disease_model.pkl')
    
    # Multiple patients
    patients_df = pd.DataFrame([
        {
            'age': 45, 'gender': 1, 'height': 175, 'weight': 80,
            'ap_hi': 120, 'ap_lo': 80, 'cholesterol': 1, 'gluc': 1,
            'smoke': 0, 'alco': 0, 'active': 1
        },
        {
            'age': 60, 'gender': 2, 'height': 160, 'weight': 85,
            'ap_hi': 160, 'ap_lo': 100, 'cholesterol': 3, 'gluc': 2,
            'smoke': 1, 'alco': 1, 'active': 0
        },
        {
            'age': 35, 'gender': 1, 'height': 180, 'weight': 75,
            'ap_hi': 110, 'ap_lo': 70, 'cholesterol': 1, 'gluc': 1,
            'smoke': 0, 'alco': 0, 'active': 1
        }
    ])
    
    # Make predictions
    results = predictor.predict(patients_df)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for i, result in enumerate(results):
        print(f"\nPatient {i+1}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability: {result['disease_probability']*100:.1f}%")
    
    return results


def interactive_prediction():
    """
    Interactive mode: Get user input and make prediction
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 80)
    
    try:
        predictor = CardiovascularPredictor('cardio_disease_model.pkl')
        
        print("\nPlease enter patient information:")
        print("-" * 80)
        
        patient = {}
        patient['age'] = int(input("Age (years): "))
        patient['gender'] = int(input("Gender (1=Male, 2=Female): "))
        patient['height'] = float(input("Height (cm): "))
        patient['weight'] = float(input("Weight (kg): "))
        patient['ap_hi'] = int(input("Systolic Blood Pressure: "))
        patient['ap_lo'] = int(input("Diastolic Blood Pressure: "))
        patient['cholesterol'] = int(input("Cholesterol (1=Normal, 2=Above, 3=Well Above): "))
        patient['gluc'] = int(input("Glucose (1=Normal, 2=Above, 3=Well Above): "))
        patient['smoke'] = int(input("Smoker (0=No, 1=Yes): "))
        patient['alco'] = int(input("Alcohol (0=No, 1=Yes): "))
        patient['active'] = int(input("Physically Active (0=No, 1=Yes): "))
        
        # Make prediction
        result = predictor.predict(patient)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n❌ Prediction cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print(" CARDIOVASCULAR DISEASE PREDICTION SYSTEM")
    print("=" * 80)
    print("\n Choose an option:")
    print("  1. Single patient prediction (example)")
    print("  2. Multiple patients prediction (example)")
    print("  3. Interactive mode (enter data manually)")
    print("  4. Batch prediction from CSV file")
    print("  0. Exit")
    print()
    
    try:
        choice = input("Enter choice (0-4): ").strip()
        
        if choice == '1':
            example_single_prediction()
            
        elif choice == '2':
            example_multiple_patients()
            
        elif choice == '3':
            interactive_prediction()
            
        elif choice == '4':
            csv_file = input("\nEnter path to CSV file: ").strip()
            output_file = input("Enter output path (or press Enter to skip): ").strip()
            output_file = output_file if output_file else None
            
            predictor = CardiovascularPredictor('cardio_disease_model.pkl')
            predictor.predict_batch(csv_file, output_file)
            
        elif choice == '0':
            print("\n👋 Goodbye!")
            
        else:
            print("\n❌ Invalid choice. Please run again and select 0-4.")
    
    except FileNotFoundError:
        print("\n❌ ERROR: Model file 'cardio_disease_model.pkl' not found!")
        print("   Please run 'train_model.py' first to train and save the model.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
