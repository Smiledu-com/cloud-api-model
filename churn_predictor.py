"""
School Churn Prediction - Production Prediction Script
This script loads a pre-trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json

class ChurnPredictor:
    def __init__(self, model_path='xgb_native_model.json', columns_path='model_columns.json'):
        """Initialize the predictor with paths to the model and column list"""
        # Load the model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load the expected columns
        with open(columns_path, 'r') as f:
            self.model_columns = json.load(f)
        print(f"Loaded {len(self.model_columns)} model columns")
        
        # Define categorical columns (needed for encoding)
        self.categorical_columns = [
            'type', 'plan', 'economy_level', 'tuition_rank', 'school_level',
            'decision_maker_role', 'geographical_region', 'contract_type',
            'payment_frequency', 'support_ticket_categories', 'academic_period',
            'budget_cycle_position'
        ]
    
    def preprocess_data(self, df):
        """Preprocess the data to match the format expected by the model"""
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Drop columns not used for prediction
        columns_to_drop = [col for col in ['school_name', 'record_date', 'churn_reason', 'churned'] 
                           if col in df_copy.columns]
        df_copy = df_copy.drop(columns=columns_to_drop, errors='ignore')
        
        # Convert categorical columns to dummy variables
        df_encoded = pd.get_dummies(df_copy, columns=[col for col in self.categorical_columns 
                                                     if col in df_copy.columns], 
                                    drop_first=True)
        
        # Ensure all expected columns exist
        for col in self.model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Select only the columns the model was trained on, in the right order
        return df_encoded[self.model_columns]
    
    def predict(self, df):
        """Make predictions on new data"""
        # Preprocess the data
        X = self.preprocess_data(df)
        
        # Make predictions
        prediction_probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (prediction_probabilities >= 0.5).astype(int)
        
        # Return both raw probabilities and binary predictions
        result_df = df.copy()
        result_df['churn_probability'] = prediction_probabilities
        result_df['predicted_churn'] = predictions
        
        return result_df
    
    def identify_high_risk_schools(self, df, threshold=0.7):
        """Identify schools with high churn risk"""
        # Get predictions
        predictions = self.predict(df)
        
        # Filter for high-risk schools
        high_risk = predictions[predictions['churn_probability'] >= threshold]
        
        # Group by school to get the latest assessment for each
        if 'school_id' in high_risk.columns:
            latest_records = high_risk.groupby('school_id').last().reset_index()
            high_risk_schools = latest_records.sort_values(by='churn_probability', ascending=False)
            
            # Select important columns for output
            output_columns = ['school_id', 'churn_probability', 'predicted_churn']
            
            # Add optional columns if they exist
            for col in ['school_name', 'churn_risk_score', 'health_score', 'tenure_weeks']:
                if col in high_risk_schools.columns:
                    output_columns.append(col)
            
            return high_risk_schools[output_columns]
        else:
            # If no school_id, just return all high-risk records
            return high_risk.sort_values(by='churn_probability', ascending=False)


# Example usage in production
if __name__ == "__main__":
    print("Churn Prediction Service")
    
    # Initialize the predictor
    predictor = ChurnPredictor()
    
    # Example: Load new data and make predictions
    # In production, this could come from your database, API, etc.
    try:
        print("Loading sample data for prediction...")
        new_data = pd.read_csv('new_school_data.csv')
        print(f"Loaded {len(new_data)} records for prediction")
        
        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(new_data)
        
        # Identify high-risk schools
        high_risk_schools = predictor.identify_high_risk_schools(new_data, threshold=0.1)
        print(f"Found {len(high_risk_schools)} high-risk schools")
        print(high_risk_schools.head())
        
        # Save results
        predictions.to_csv('predictions.csv', index=False)
        high_risk_schools.to_csv('high_risk_schools.csv', index=False)
        print("Results saved to 'predictions.csv' and 'high_risk_schools.csv'")
        
    except FileNotFoundError:
        print("No new data file found. This script is ready to use with your real data.")
        print("In production, you would connect this to your data pipeline or API.")