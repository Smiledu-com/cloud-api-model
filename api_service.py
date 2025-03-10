"""
School Churn Prediction - REST API Service
This script serves the churn prediction model as a REST API using FastAPI.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import uvicorn
from churn_predictor import ChurnPredictor

# Initialize the FastAPI app
app = FastAPI(
    title="School Churn Prediction API",
    description="API for predicting school churn risk",
    version="1.0.0"
)

# Initialize the predictor
model_path = os.environ.get("MODEL_PATH", "xgb_native_model.json")
columns_path = os.environ.get("COLUMNS_PATH", "model_columns.json")
predictor = ChurnPredictor(model_path=model_path, columns_path=columns_path)

# Define input data models
class SchoolFeatures(BaseModel):
    """Input features for a school record"""
    school_id: int
    school_name: str
    type: str  # public/private/charter
    plan: str  # starter/premium/enterprise
    economy_level: str
    number_students: int
    tuition_rank: str
    school_level: str
    decision_maker_role: str
    geographical_region: str
    tenure_weeks: int
    contract_remaining_weeks: int
    contract_type: str
    recent_contract_change: int
    renewal_count: int
    payment_delay: int
    payment_frequency: str
    recent_price_change: float
    price_sensitivity: float
    discount_applied: int
    active_users_percentage: float
    weekly_active_users: int
    usage_trend: float
    feature_adoption_rate: float
    critical_feature_usage: int
    peak_usage_day: int
    session_duration_avg: int
    session_frequency: float
    idle_user_percentage: float
    usage_consistency: float
    number_messages_to_support: int
    complain_level: float
    support_response_time: float
    support_ticket_trend: float
    tech_savy_level: float
    number_complaints_with_ceo: int
    support_ticket_categories: str
    sentiment: float
    nps_score: int
    nps_trend: float
    sentiment_trend: float
    feedback_participation: float
    positive_feedback_ratio: float
    feature_satisfaction: int
    training_satisfaction: int
    bugs_ratio: float
    ratio_usability: float
    error_rate: float
    page_load_time: int
    historic_complain_ratio: float
    critical_bug_count: int
    workflow_completion_rate: float
    ratio_feature_requests: float
    feature_request_fulfillment: float
    days_since_feature_update: int
    alignment_with_roadmap: float
    competitor_feature_mentions: int
    days_since_admin_login: int
    stakeholder_engagement: float
    training_completion_rate: float
    days_since_cs_contact: int
    meeting_attendance_rate: float
    proactive_contact_ratio: float
    week_of_year: int
    academic_period: str
    days_to_break: int
    budget_cycle_position: str
    is_renewal_period: int
    churn_risk_score: float
    usage_to_complaint_ratio: float
    engagement_trend: float
    health_score: float
    relative_performance: float
    roi_indicator: float
    feature_value_ratio: float
    
    class Config:
        schema_extra = {
            "example": {
                "school_id": 1001,
                "school_name": "Example School",
                "type": "private",
                "plan": "premium",
                "economy_level": "medium resource",
                "number_students": 1500,
                "tuition_rank": "high",
                "school_level": "k-12",
                "decision_maker_role": "IT Director",
                "geographical_region": "Northeast",
                "tenure_weeks": 45,
                "contract_remaining_weeks": 30,
                "contract_type": "annual",
                "recent_contract_change": 0,
                "renewal_count": 0,
                "payment_delay": 0,
                "payment_frequency": "quarterly",
                "recent_price_change": 0.0,
                "price_sensitivity": 0.47,
                "discount_applied": 0,
                "active_users_percentage": 84.3,
                "weekly_active_users": 450,
                "usage_trend": 0.12,
                "feature_adoption_rate": 82.4,
                "critical_feature_usage": 1,
                "peak_usage_day": 3,
                "session_duration_avg": 37,
                "session_frequency": 8.6,
                "idle_user_percentage": 15.7,
                "usage_consistency": 0.85,
                "number_messages_to_support": 2,
                "complain_level": 0.09,
                "support_response_time": 5.7,
                "support_ticket_trend": -0.12,
                "tech_savy_level": 0.78,
                "number_complaints_with_ceo": 0,
                "support_ticket_categories": "login",
                "sentiment": 0.87,
                "nps_score": 9,
                "nps_trend": 0.3,
                "sentiment_trend": 0.07,
                "feedback_participation": 58.7,
                "positive_feedback_ratio": 0.82,
                "feature_satisfaction": 4,
                "training_satisfaction": 4,
                "bugs_ratio": 0.04,
                "ratio_usability": 0.07,
                "error_rate": 4.0,
                "page_load_time": 724,
                "historic_complain_ratio": 0.11,
                "critical_bug_count": 0,
                "workflow_completion_rate": 92.3,
                "ratio_feature_requests": 0.12,
                "feature_request_fulfillment": 0.52,
                "days_since_feature_update": 12,
                "alignment_with_roadmap": 0.83,
                "competitor_feature_mentions": 0,
                "days_since_admin_login": 2,
                "stakeholder_engagement": 0.85,
                "training_completion_rate": 0.93,
                "days_since_cs_contact": 5,
                "meeting_attendance_rate": 0.95,
                "proactive_contact_ratio": 0.72,
                "week_of_year": 10,
                "academic_period": "winter mid-term",
                "days_to_break": 47,
                "budget_cycle_position": "allocated",
                "is_renewal_period": 0,
                "churn_risk_score": 0.25,
                "usage_to_complaint_ratio": 9.37,
                "engagement_trend": 0.07,
                "health_score": 1.73,
                "relative_performance": 0.83,
                "roi_indicator": 0.75,
                "feature_value_ratio": 0.64
            }
        }

class SchoolBatch(BaseModel):
    """Batch of school records"""
    schools: List[SchoolFeatures]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    school_id: int
    school_name: str
    churn_probability: float
    predicted_churn: int
    high_risk: bool

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    high_risk_count: int
    total_schools: int

# API endpoints
@app.get("/")
async def root():
    return {"message": "School Churn Prediction API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(school: SchoolFeatures, threshold: float = Query(0.3, description="Churn probability threshold for high risk")):
    """
    Predict churn probability for a single school
    """
    try:
        # Convert Pydantic model to pandas DataFrame
        df = pd.DataFrame([school.dict()])
        
        # Make prediction
        result = predictor.predict(df)
        
        # Extract result
        prediction = result.iloc[0]
        
        return {
            "school_id": int(prediction["school_id"]),
            "school_name": prediction["school_name"],
            "churn_probability": float(prediction["churn_probability"]),
            "predicted_churn": int(prediction["predicted_churn"]),
            "high_risk": bool(prediction["churn_probability"] >= threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: SchoolBatch, threshold: float = Query(0.3, description="Churn probability threshold for high risk")):
    """
    Predict churn probability for a batch of schools
    """
    try:
        # Convert batch to DataFrame
        schools_data = [school.dict() for school in batch.schools]
        df = pd.DataFrame(schools_data)
        
        # Make predictions
        results = predictor.predict(df)
        
        # Format response
        predictions = []
        for _, row in results.iterrows():
            predictions.append({
                "school_id": int(row["school_id"]),
                "school_name": row["school_name"],
                "churn_probability": float(row["churn_probability"]),
                "predicted_churn": int(row["predicted_churn"]),
                "high_risk": bool(row["churn_probability"] >= threshold)
            })
        
        high_risk_count = sum(1 for p in predictions if p["high_risk"])
        
        return {
            "predictions": predictions,
            "high_risk_count": high_risk_count,
            "total_schools": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/identify_high_risk")
async def identify_high_risk(batch: SchoolBatch, threshold: float = Query(0.3, description="Churn probability threshold for high risk")):
    """
    Identify high-risk schools from a batch
    """
    try:
        # Convert batch to DataFrame
        schools_data = [school.dict() for school in batch.schools]
        df = pd.DataFrame(schools_data)
        
        # Identify high-risk schools
        high_risk = predictor.identify_high_risk_schools(df, threshold=threshold)
        
        # Convert to response format
        if high_risk.empty:
            return {"message": "No high-risk schools found", "schools": []}
        
        high_risk_list = high_risk.to_dict(orient="records")
        return {"message": f"Found {len(high_risk_list)} high-risk schools", "schools": high_risk_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"High risk identification error: {str(e)}")

# Run the API server when the script is executed
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_service:app", host="0.0.0.0", port=port, reload=False)