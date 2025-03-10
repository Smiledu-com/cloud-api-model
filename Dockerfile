FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the model and code
COPY school_churn_model.json .
COPY model_columns.json .
COPY churn_predictor.py .
COPY api_service.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python3", "api_service.py"]