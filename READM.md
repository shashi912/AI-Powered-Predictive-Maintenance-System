# AI-Powered Predictive Maintenance System

## Overview
This project predicts machine failures in manufacturing industries using AI. 
It allows proactive maintenance, reduces downtime, and provides actionable insights via a dashboard.

## Features
- Predict equipment failure (Yes/No)
- Dashboard for visualizing predictions
- Sample predictions using sensor data

## Folder Structure
- `data/` : Dataset
- `notebooks/` : EDA and model training
- `src/` : Source code
- `outputs/` : Saved model and sample predictions

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/model.py`
3. Run dashboard: `streamlit run src/dashboard.py`
