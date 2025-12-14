#!/usr/bin/env python3

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from clearml import Task, Logger, OutputModel
from clearml.automation import (
    UniformParameterRange,
    UniformIntegerParameterRange,
    DiscreteParameterRange,
    HyperParameterOptimizer,
    RandomSearch,
    GridSearch
)

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def load_dataset(data_path: str = None) -> pd.DataFrame:
    if data_path is None:
        data_path = DATA_DIR / "weather_dataset.csv"
    
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude_cols = [
        "date", "city", 
        "target_d1", "target_d2", "target_d3", "target_d4", 
        "target_d5", "target_d6", "target_d7",
        "temp_avg", "temp_max", "temp_min"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_and_evaluate(
    df: pd.DataFrame,
    params: Dict,
    target_col: str = "target_d1",
    n_splits: int = 3
) -> Tuple[float, Dict[str, float]]:
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[target_col]
    
    df_sorted = df.sort_values("date")
    sorted_indices = df_sorted.index.tolist()
    
    X = X.loc[sorted_indices]
    y = y.loc[sorted_indices]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        train_params = params.copy()
        n_estimators = train_params.pop("n_estimators", 500)
        early_stopping = train_params.pop("early_stopping_rounds", 50)
        
        model = lgb.train(
            train_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(early_stopping),
                lgb.log_evaluation(period=0)
            ]
        )
        
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val.values, y_pred)
        fold_metrics.append(metrics)
    
    avg_metrics = {
        "mae": np.mean([m["mae"] for m in fold_metrics]),
        "rmse": np.mean([m["rmse"] for m in fold_metrics]),
        "mape": np.mean([m["mape"] for m in fold_metrics])
    }
    
    return avg_metrics["rmse"], avg_metrics


def run_grid_search(
    df: pd.DataFrame,
    param_grid: Dict[str, List],
    target_col: str = "target_d1",
    task: Task = None
) -> Tuple[Dict, Dict[str, float], pd.DataFrame]:
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\n🔍 Grid Search: {len(combinations)} конфигураций")
    
    results = []
    best_rmse = float("inf")
    best_params = None
    best_metrics = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        full_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50,
            **params
        }
        
        print(f"\n📊 Конфигурация {i+1}/{len(combinations)}:")
        print(f"   {params}")
        
        try:
            rmse, metrics = train_and_evaluate(df, full_params, target_col)
            
            print(f"   RMSE: {rmse:.4f}, MAE: {metrics['mae']:.4f}")
            
            result = {
                "config_id": i + 1,
                **params,
                **{f"val_{k}": v for k, v in metrics.items()}
            }
            results.append(result)
            
            if task:
                logger = task.get_logger()
                for metric_name, value in metrics.items():
                    logger.report_scalar(
                        title="HPO Metrics",
                        series=metric_name,
                        value=value,
                        iteration=i
                    )
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = full_params.copy()
                best_metrics = metrics.copy()
                print(f"   ✅ Новый лучший результат!")
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results.append({
                "config_id": i + 1,
                **params,
                "error": str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    return best_params, best_metrics, results_df


def register_best_model(
    df: pd.DataFrame,
    best_params: Dict,
    project_name: str = "Weather Forecast",
    model_name: str = "weather_model",
    version: str = None,
    tags: List[str] = None,
    task: Task = None
) -> str:
    import pickle
    
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if tags is None:
        tags = ["production", "best_hpo"]
    
    print(f"\n🏆 Обучение финальной модели с лучшими параметрами...")
    
    if task is None:
        try:
            current_task = Task.current_task()
            if current_task:
                task = current_task
            else:
                task = Task.init(
                    project_name=project_name,
                    task_name=f"Best Model Training - {version}",
                    task_type=Task.TaskTypes.training
                )
        except:
            task = Task.init(
                project_name=project_name,
                task_name=f"Best Model Training - {version}",
                task_type=Task.TaskTypes.training
            )
    
    task.connect(best_params)
    
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    models = {}
    
    for h in range(1, 8):
        target_col = f"target_d{h}"
        
        if target_col not in df.columns:
            continue
        
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col]
        
        train_params = best_params.copy()
        n_estimators = train_params.pop("n_estimators", 500)
        early_stopping = train_params.pop("early_stopping_rounds", 50)
        
        train_data = lgb.Dataset(X, label=y)
        
        model = lgb.train(
            train_params,
            train_data,
            num_boost_round=n_estimators
        )
        
        models[target_col] = model
        print(f"   ✅ Модель {target_col} обучена")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"weather_ensemble_{version}.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": models,
            "params": best_params,
            "feature_columns": numeric_cols,
            "version": version,
            "created_at": datetime.now().isoformat()
        }, f)
    
    print(f"💾 Модель сохранена: {model_path}")
    
    output_model = OutputModel(
        task=task,
        framework="lightgbm",
        name=model_name
    )
    
    output_model.update_weights(
        weights_filename=str(model_path),
        auto_delete_file=False
    )
    
    output_model.update_design(config_dict={
        "version": version,
        "best_params": best_params,
        "horizons": list(models.keys()),
        "feature_columns": numeric_cols
    })
    
    for tag in tags:
        task.add_tags([tag])
    
    print(f"📤 Модель зарегистрирована в ClearML Model Registry")
    print(f"   Версия: {version}")
    print(f"   Теги: {tags}")
    
    return str(model_path)


def main():
    print("=" * 60)
    print("🔍 Поиск гиперпараметров (HPO)")
    print("=" * 60)
    
    try:
        task = Task.init(
            project_name="Weather Forecast",
            task_name="Hyperparameter Optimization",
            task_type=Task.TaskTypes.optimizer
        )
        print(f"📋 ClearML Task ID: {task.id}")
    except Exception as e:
        print(f"⚠️ ClearML не доступен: {e}")
        task = None
    
    print("\n📊 Загрузка данных...")
    df = load_dataset()
    print(f"   Загружено {len(df)} записей")
    
    quick_param_grid = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "feature_fraction": [0.8],
        "bagging_fraction": [0.8],
    }
    
    print(f"\n📋 Сетка параметров:")
    for param, values in quick_param_grid.items():
        print(f"   {param}: {values}")
    
    best_params, best_metrics, results_df = run_grid_search(
        df, 
        quick_param_grid,
        target_col="target_d1",
        task=task
    )
    
    extra_params = {
        "num_leaves": [47],
        "learning_rate": [0.03],
        "feature_fraction": [0.85],
        "bagging_fraction": [0.85],
    }
    
    _, extra_metrics, extra_results = run_grid_search(
        df,
        extra_params,
        target_col="target_d1",
        task=task
    )
    
    results_df = pd.concat([results_df, extra_results], ignore_index=True)
    
    print("\n" + "=" * 60)
    print("📊 Результаты HPO:")
    print("=" * 60)
    print(f"\n🏆 Лучшие параметры:")
    for k, v in best_params.items():
        if k not in ["objective", "metric", "boosting_type", "verbose"]:
            print(f"   {k}: {v}")
    
    print(f"\n📈 Лучшие метрики:")
    print(f"   RMSE: {best_metrics['rmse']:.4f}")
    print(f"   MAE: {best_metrics['mae']:.4f}")
    print(f"   MAPE: {best_metrics['mape']:.2f}%")
    
    results_path = MODELS_DIR / "hpo_results.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Результаты сохранены: {results_path}")
    
    if task:
        task.get_logger().report_table(
            title="HPO Results",
            series="All Configurations",
            table_plot=results_df
        )
        
        task.upload_artifact(
            name="best_params",
            artifact_object=best_params
        )
    
    print("\n" + "=" * 60)
    model_path = register_best_model(
        df,
        best_params,
        tags=["production", "hpo_best", f"rmse_{best_metrics['rmse']:.3f}"],
        task=task
    )
    
    if task:
        task.close()
    
    print("\n" + "=" * 60)
    print("✅ HPO завершен!")
    print(f"📁 Модель: {model_path}")
    print(f"📊 Результаты: {results_path}")
    print("=" * 60)
    
    return best_params, best_metrics, results_df


if __name__ == "__main__":
    main()
