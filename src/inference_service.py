#!/usr/bin/env python3

import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "москва": {"latitude": 55.7558, "longitude": 37.6173, "name": "Москва"},
    "санкт-петербург": {"latitude": 59.9343, "longitude": 30.3351, "name": "Санкт-Петербург"},
    "moscow": {"latitude": 55.7558, "longitude": 37.6173, "name": "Москва"},
    "saint-petersburg": {"latitude": 59.9343, "longitude": 30.3351, "name": "Санкт-Петербург"},
}

app = FastAPI(
    title="Weather Forecast API",
    description="API для прогноза температуры на 7 дней вперед",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    city: str = Field(..., description="Название города (Москва, Санкт-Петербург)")
    dates: Optional[List[str]] = Field(None, description="Список дат для прогноза (формат YYYY-MM-DD)")
    additional_features: Optional[Dict[str, float]] = Field(None, description="Дополнительные признаки")


class DayForecast(BaseModel):
    date: str
    day_offset: int
    predicted_temp_avg: float
    confidence_interval: Dict[str, float]
    prediction_std: Optional[float] = None


class PredictResponse(BaseModel):
    city: str
    request_date: str
    forecasts: List[DayForecast]
    model_version: Optional[str] = None
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    available_cities: List[str]


model_data = None
model_version = None


def load_model():
    global model_data, model_version
    
    model_files = list(MODELS_DIR.glob("weather_ensemble*.pkl"))
    
    if not model_files:
        print("⚠️ Модель не найдена, используется заглушка")
        model_data = None
        model_version = "stub"
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_model, "rb") as f:
        model_data = pickle.load(f)
    
    model_version = model_data.get("version", latest_model.stem)
    print(f"✅ Модель загружена: {latest_model.name}")
    print(f"   Версия: {model_version}")


def get_historical_features(
    city: str,
    target_date: datetime,
    coords: Dict[str, float]
) -> Dict[str, float]:
    end_date = target_date - timedelta(days=1)
    start_date = end_date - timedelta(days=35)
    
    try:
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
            "timezone": "Europe/Moscow"
        }
        
        response = requests.get(OPEN_METEO_HISTORICAL_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        temps_avg = daily.get("temperature_2m_mean", [])
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        
        if not temps_avg:
            raise ValueError("Нет данных о температуре")
        
        features = {
            "temp_avg_lag_1": temps_avg[-1] if len(temps_avg) >= 1 else 0,
            "temp_avg_lag_2": temps_avg[-2] if len(temps_avg) >= 2 else 0,
            "temp_avg_lag_3": temps_avg[-3] if len(temps_avg) >= 3 else 0,
            "temp_avg_lag_7": temps_avg[-7] if len(temps_avg) >= 7 else 0,
            "temp_avg_lag_14": temps_avg[-14] if len(temps_avg) >= 14 else 0,
            
            "temp_max_lag_1": temps_max[-1] if len(temps_max) >= 1 else 0,
            "temp_max_lag_2": temps_max[-2] if len(temps_max) >= 2 else 0,
            "temp_max_lag_3": temps_max[-3] if len(temps_max) >= 3 else 0,
            "temp_max_lag_7": temps_max[-7] if len(temps_max) >= 7 else 0,
            "temp_max_lag_14": temps_max[-14] if len(temps_max) >= 14 else 0,
            
            "temp_min_lag_1": temps_min[-1] if len(temps_min) >= 1 else 0,
            "temp_min_lag_2": temps_min[-2] if len(temps_min) >= 2 else 0,
            "temp_min_lag_3": temps_min[-3] if len(temps_min) >= 3 else 0,
            "temp_min_lag_7": temps_min[-7] if len(temps_min) >= 7 else 0,
            "temp_min_lag_14": temps_min[-14] if len(temps_min) >= 14 else 0,
            
            "precipitation_lag_1": precip[-1] if len(precip) >= 1 else 0,
            "precipitation_lag_2": precip[-2] if len(precip) >= 2 else 0,
            "precipitation_lag_3": precip[-3] if len(precip) >= 3 else 0,
            "precipitation_lag_7": precip[-7] if len(precip) >= 7 else 0,
            
            "temp_avg_rolling_7": np.mean(temps_avg[-7:]) if len(temps_avg) >= 7 else np.mean(temps_avg),
            "temp_avg_rolling_14": np.mean(temps_avg[-14:]) if len(temps_avg) >= 14 else np.mean(temps_avg),
            "temp_avg_rolling_30": np.mean(temps_avg[-30:]) if len(temps_avg) >= 30 else np.mean(temps_avg),
            
            "temp_max_rolling_7": np.max(temps_max[-7:]) if len(temps_max) >= 7 else np.max(temps_max),
            "temp_max_rolling_14": np.max(temps_max[-14:]) if len(temps_max) >= 14 else np.max(temps_max),
            "temp_max_rolling_30": np.max(temps_max[-30:]) if len(temps_max) >= 30 else np.max(temps_max),
            
            "temp_min_rolling_7": np.min(temps_min[-7:]) if len(temps_min) >= 7 else np.min(temps_min),
            "temp_min_rolling_14": np.min(temps_min[-14:]) if len(temps_min) >= 14 else np.min(temps_min),
            "temp_min_rolling_30": np.min(temps_min[-30:]) if len(temps_min) >= 30 else np.min(temps_min),
            
            "precipitation_rolling_7": np.sum(precip[-7:]) if len(precip) >= 7 else np.sum(precip),
            "precipitation_rolling_14": np.sum(precip[-14:]) if len(precip) >= 14 else np.sum(precip),
            "precipitation_rolling_30": np.sum(precip[-30:]) if len(precip) >= 30 else np.sum(precip),
            
            "temp_change_1d": temps_avg[-1] - temps_avg[-2] if len(temps_avg) >= 2 else 0,
            "temp_change_7d": temps_avg[-1] - temps_avg[-7] if len(temps_avg) >= 7 else 0,
            
            "temp_range": temps_max[-1] - temps_min[-1] if (len(temps_max) >= 1 and len(temps_min) >= 1) else 0,
            
            "has_precipitation": 1 if (len(precip) >= 1 and precip[-1] > 0) else 0,
        }
        
        return features
    
    except Exception as e:
        print(f"⚠️ Ошибка получения исторических данных: {e}")
        return {
            "temp_avg_lag_1": 10.0, "temp_avg_lag_2": 10.0, "temp_avg_lag_3": 10.0,
            "temp_avg_lag_7": 10.0, "temp_avg_lag_14": 10.0,
            "temp_max_lag_1": 15.0, "temp_max_lag_2": 15.0, "temp_max_lag_3": 15.0,
            "temp_max_lag_7": 15.0, "temp_max_lag_14": 15.0,
            "temp_min_lag_1": 5.0, "temp_min_lag_2": 5.0, "temp_min_lag_3": 5.0,
            "temp_min_lag_7": 5.0, "temp_min_lag_14": 5.0,
            "precipitation_lag_1": 0, "precipitation_lag_2": 0,
            "precipitation_lag_3": 0, "precipitation_lag_7": 0,
            "temp_avg_rolling_7": 10.0, "temp_avg_rolling_14": 10.0, "temp_avg_rolling_30": 10.0,
            "temp_max_rolling_7": 15.0, "temp_max_rolling_14": 15.0, "temp_max_rolling_30": 15.0,
            "temp_min_rolling_7": 5.0, "temp_min_rolling_14": 5.0, "temp_min_rolling_30": 5.0,
            "precipitation_rolling_7": 0, "precipitation_rolling_14": 0, "precipitation_rolling_30": 0,
            "temp_change_1d": 0, "temp_change_7d": 0, "temp_range": 10.0, "has_precipitation": 0
        }


def create_calendar_features(target_date: datetime) -> Dict[str, float]:
    day_of_year = target_date.timetuple().tm_yday
    day_of_week = target_date.weekday()
    
    return {
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "month": target_date.month,
        "year": target_date.year,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365),
        "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
    }


def predict_stub(
    city: str,
    dates: List[datetime],
    coords: Dict[str, float]
) -> List[DayForecast]:
    forecasts = []
    base_date = datetime.now().date()
    
    for i, date in enumerate(dates):
        day_offset = (date.date() - base_date).days
        
        day_of_year = date.timetuple().tm_yday
        
        avg_temps = [-8, -7, -1, 7, 13, 17, 20, 18, 12, 6, -1, -6]
        month_idx = date.month - 1
        base_temp = avg_temps[month_idx]
        
        temp_variation = np.sin(2 * np.pi * day_of_year / 365) * 5
        predicted_temp = base_temp + temp_variation + np.random.normal(0, 2)
        
        forecasts.append(DayForecast(
            date=date.strftime("%Y-%m-%d"),
            day_offset=day_offset,
            predicted_temp_avg=round(predicted_temp, 1),
            confidence_interval={
                "lower": round(predicted_temp - 3.0, 1),
                "upper": round(predicted_temp + 3.0, 1)
            },
            prediction_std=3.0
        ))
    
    return forecasts


def predict_with_model(
    city: str,
    dates: List[datetime],
    coords: Dict[str, float],
    additional_features: Optional[Dict[str, float]] = None
) -> List[DayForecast]:
    if model_data is None:
        return predict_stub(city, dates, coords)
    
    models = model_data.get("models", {})
    feature_columns = model_data.get("feature_columns", [])
    
    if not models:
        return predict_stub(city, dates, coords)
    
    forecasts = []
    base_date = datetime.now().date()
    
    historical_features = get_historical_features(city, datetime.now(), coords)
    
    for i, date in enumerate(dates):
        day_offset = (date.date() - base_date).days
        target_key = f"target_d{day_offset}"
        
        if target_key not in models:
            available_horizons = [int(k.split("_d")[1]) for k in models.keys()]
            closest = min(available_horizons, key=lambda x: abs(x - day_offset))
            target_key = f"target_d{closest}"
        
        model = models[target_key]
        
        calendar_features = create_calendar_features(date)
        
        all_features = {
            **historical_features,
            **calendar_features,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
        }
        
        if additional_features:
            all_features.update(additional_features)
        
        feature_values = []
        for col in feature_columns:
            if col in all_features:
                feature_values.append(all_features[col])
            else:
                feature_values.append(0)
        
        X = np.array([feature_values])
        
        predicted_temp = model.predict(X)[0]
        
        base_std = 1.5
        std = base_std * (1 + 0.2 * day_offset)
        
        forecasts.append(DayForecast(
            date=date.strftime("%Y-%m-%d"),
            day_offset=day_offset,
            predicted_temp_avg=round(float(predicted_temp), 1),
            confidence_interval={
                "lower": round(float(predicted_temp - 1.96 * std), 1),
                "upper": round(float(predicted_temp + 1.96 * std), 1)
            },
            prediction_std=round(std, 2)
        ))
    
    return forecasts


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_data is not None,
        model_version=model_version,
        available_cities=list(CITIES.keys())
    )


@app.get("/")
async def root():
    return {
        "service": "Weather Forecast API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    city_key = request.city.lower().strip()
    
    if city_key not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный город: {request.city}. Доступные города: {list(CITIES.keys())}"
        )
    
    coords = CITIES[city_key]
    city_name = coords["name"]
    
    today = datetime.now().date()
    
    if request.dates:
        try:
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in request.dates]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Неверный формат даты. Используйте YYYY-MM-DD. Ошибка: {e}"
            )
    else:
        dates = [datetime.combine(today + timedelta(days=i), datetime.min.time()) for i in range(1, 8)]
    
    forecasts = predict_with_model(
        city_name,
        dates,
        coords,
        request.additional_features
    )
    
    return PredictResponse(
        city=city_name,
        request_date=today.strftime("%Y-%m-%d"),
        forecasts=forecasts,
        model_version=model_version,
        metadata={
            "coordinates": {"latitude": coords["latitude"], "longitude": coords["longitude"]},
            "forecast_horizon_days": len(forecasts)
        }
    )


@app.get("/cities")
async def get_cities():
    return {
        "cities": [
            {"key": key, "name": value["name"], "coordinates": {"latitude": value["latitude"], "longitude": value["longitude"]}}
            for key, value in CITIES.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
