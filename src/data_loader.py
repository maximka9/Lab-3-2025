#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests
from clearml import Dataset, Task
from dotenv import load_dotenv

load_dotenv()

OPEN_METEO_HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "Москва": {"latitude": 55.7558, "longitude": 37.6173},
    "Санкт-Петербург": {"latitude": 59.9343, "longitude": 30.3351},
}

END_DATE = datetime.now().date() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=3*365 + 60)

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_historical_weather_chunk(
    city_name: str,
    coordinates: Dict[str, float],
    start_date: datetime,
    end_date: datetime,
    max_retries: int = 3
) -> pd.DataFrame:
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration"
        ]),
        "timezone": "Europe/Moscow"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                OPEN_METEO_HISTORICAL_API, 
                params=params, 
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            df = pd.DataFrame({
                "date": pd.to_datetime(daily.get("time", [])),
                "city": city_name,
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "temp_avg": daily.get("temperature_2m_mean", []),
                "precipitation_sum": daily.get("precipitation_sum", []),
                "rain_sum": daily.get("rain_sum", []),
                "snowfall_sum": daily.get("snowfall_sum", []),
                "precipitation_hours": daily.get("precipitation_hours", []),
                "wind_speed_max": daily.get("wind_speed_10m_max", []),
                "wind_gusts_max": daily.get("wind_gusts_10m_max", []),
                "wind_direction_dominant": daily.get("wind_direction_10m_dominant", []),
                "shortwave_radiation_sum": daily.get("shortwave_radiation_sum", []),
                "evapotranspiration": daily.get("et0_fao_evapotranspiration", [])
            })
            
            return df
        
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"   ⚠️ Таймаут, повторная попытка через {wait_time} сек... (попытка {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Не удалось загрузить данные после {max_retries} попыток: {e}")
    
    return pd.DataFrame()


def fetch_historical_weather(
    city_name: str, 
    coordinates: Dict[str, float],
    start_date,
    end_date
) -> pd.DataFrame:
    from datetime import date as date_type
    if type(start_date) is date_type:
        start_date = datetime.combine(start_date, datetime.min.time())
    if type(end_date) is date_type:
        end_date = datetime.combine(end_date, datetime.min.time())
    
    print(f"📥 Загрузка данных для {city_name} ({start_date.date()} - {end_date.date()})...")
    
    all_chunks = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(
            datetime(current_start.year + 1, 1, 1),
            end_date
        )
        
        if current_end > current_start:
            print(f"   Загрузка периода: {current_start.date()} - {current_end.date()}...")
            chunk_df = fetch_historical_weather_chunk(
                city_name, 
                coordinates, 
                current_start, 
                current_end
            )
            
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
                print(f"   ✅ Загружено {len(chunk_df)} записей")
            
            time.sleep(1)
        
        current_start = current_end
    
    if not all_chunks:
        print(f"   ⚠️ Не удалось загрузить данные для {city_name}")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_chunks, ignore_index=True)
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset=["date", "city"], keep="first")
    
    print(f"   ✅ Всего загружено {len(combined_df)} записей")
    return combined_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    for lag in [1, 2, 3, 7, 14]:
        df[f"temp_avg_lag_{lag}"] = df.groupby("city")["temp_avg"].shift(lag)
        df[f"temp_max_lag_{lag}"] = df.groupby("city")["temp_max"].shift(lag)
        df[f"temp_min_lag_{lag}"] = df.groupby("city")["temp_min"].shift(lag)
    
    for lag in [1, 2, 3, 7]:
        df[f"precipitation_lag_{lag}"] = df.groupby("city")["precipitation_sum"].shift(lag)
    
    for window in [7, 14, 30]:
        df[f"temp_avg_rolling_{window}"] = df.groupby("city")["temp_avg"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"temp_max_rolling_{window}"] = df.groupby("city")["temp_max"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
        df[f"temp_min_rolling_{window}"] = df.groupby("city")["temp_min"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        )
    
    for window in [7, 14, 30]:
        df[f"precipitation_rolling_{window}"] = df.groupby("city")["precipitation_sum"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
    
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["temp_change_1d"] = df.groupby("city")["temp_avg"].diff(1)
    df["temp_change_7d"] = df.groupby("city")["temp_avg"].diff(7)
    
    df["has_precipitation"] = (df["precipitation_sum"] > 0).astype(int)
    df["has_rain"] = (df["rain_sum"] > 0).astype(int)
    df["has_snow"] = (df["snowfall_sum"] > 0).astype(int)
    
    return df


def create_target_columns(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = df.copy()
    
    for h in range(1, horizon + 1):
        df[f"target_d{h}"] = df.groupby("city")["temp_avg"].shift(-h)
    
    return df


def load_all_data(
    cities: Dict[str, Dict[str, float]] = None,
    start_date: datetime = None,
    end_date: datetime = None
) -> pd.DataFrame:
    if cities is None:
        cities = CITIES
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    
    all_data = []
    for city_name, coords in cities.items():
        df = fetch_historical_weather(city_name, coords, start_date, end_date)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Всего загружено записей: {len(combined_df)}")
    print(f"   Период: {combined_df['date'].min()} - {combined_df['date'].max()}")
    print(f"   Города: {combined_df['city'].unique().tolist()}")
    
    return combined_df


def prepare_dataset(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    print("\n🔧 Создание признаков...")
    df = create_features(df)
    
    print("🎯 Создание целевых переменных...")
    df = create_target_columns(df, horizon)
    
    target_cols = [f"target_d{h}" for h in range(1, horizon + 1)]
    initial_len = len(df)
    df = df.dropna(subset=target_cols)
    print(f"   Удалено {initial_len - len(df)} строк с пропущенными целевыми значениями")
    
    lag_cols = [col for col in df.columns if "lag_" in col or "rolling_" in col]
    df = df.dropna(subset=lag_cols)
    
    print(f"\n✅ Итоговый размер датасета: {len(df)} строк, {len(df.columns)} признаков")
    
    return df


def upload_to_clearml(
    df: pd.DataFrame,
    project_name: str = "Weather Forecast",
    dataset_name: str = "weather_historical_data",
    dataset_version: str = None
) -> Dataset:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = DATA_DIR / "weather_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Датасет сохранен локально: {csv_path}")
    
    metadata = {
        "cities": df["city"].unique().tolist(),
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max())
        },
        "num_records": len(df),
        "num_features": len(df.columns),
        "target_columns": [col for col in df.columns if col.startswith("target_")],
        "feature_columns": [col for col in df.columns if not col.startswith("target_")],
        "created_at": datetime.now().isoformat()
    }
    
    metadata_path = DATA_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    
    stats = df.describe().to_dict()
    stats_path = DATA_DIR / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    print("📤 Загрузка в ClearML...")
    
    try:
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=project_name,
            dataset_version=dataset_version
        )
        
        dataset.add_files(str(DATA_DIR))
        dataset.set_metadata(metadata)
        dataset.upload()
        dataset.finalize()
        
        print(f"✅ Датасет загружен в ClearML:")
        print(f"   ID: {dataset.id}")
        print(f"   Название: {dataset_name}")
        print(f"   Проект: {project_name}")
        
        return dataset
    
    except Exception as e:
        print(f"⚠️ Ошибка загрузки в ClearML: {e}")
        print("   Датасет сохранен локально и может быть загружен позже.")
        return None


def main():
    print("=" * 60)
    print("📊 Загрузка исторических погодных данных")
    print("=" * 60)
    print(f"Источник: Open-Meteo Historical Weather API")
    print(f"Период: {START_DATE} - {END_DATE}")
    print(f"Города: {list(CITIES.keys())}")
    print("=" * 60)
    
    raw_df = load_all_data()
    prepared_df = prepare_dataset(raw_df)
    dataset = upload_to_clearml(prepared_df)
    
    print("\n" + "=" * 60)
    print("✅ Загрузка данных завершена!")
    print("=" * 60)
    
    return prepared_df


if __name__ == "__main__":
    main()
