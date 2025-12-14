#!/usr/bin/env python3

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from clearml import Task, Logger, OutputModel

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
    
    print(f"📊 Загружен датасет: {len(df)} записей")
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
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


def train_model(
    df: pd.DataFrame,
    target_col: str = "target_d1",
    params: Dict = None,
    test_size: float = 0.2,
    random_state: int = 42,
    task: Task = None
) -> Tuple[lgb.Booster, Dict[str, float], pd.DataFrame]:
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 500,
            "early_stopping_rounds": 50
        }
    
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols].copy()
    y = df[target_col].copy()
    
    X = X.fillna(X.median())
    
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]
    
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"\n📈 Обучение модели для {target_col}")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Признаков: {len(numeric_cols)}")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    n_estimators = params.pop("n_estimators", 500)
    early_stopping = params.pop("early_stopping_rounds", 50)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(period=100)
        ]
    )
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train.values, y_pred_train)
    test_metrics = calculate_metrics(y_test.values, y_pred_test)
    
    print(f"\n📊 Результаты для {target_col}:")
    print(f"   Train MAE: {train_metrics['mae']:.3f}, RMSE: {train_metrics['rmse']:.3f}")
    print(f"   Test MAE: {test_metrics['mae']:.3f}, RMSE: {test_metrics['rmse']:.3f}")
    
    importance_df = pd.DataFrame({
        "feature": numeric_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)
    
    if task:
        logger = task.get_logger()
        
        for metric_name, value in test_metrics.items():
            logger.report_scalar(
                title=f"Metrics/{target_col}",
                series=f"test_{metric_name}",
                value=value,
                iteration=0
            )
        
        for metric_name, value in train_metrics.items():
            logger.report_scalar(
                title=f"Metrics/{target_col}",
                series=f"train_{metric_name}",
                value=value,
                iteration=0
            )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(20)
        ax.barh(top_features["feature"], top_features["importance"])
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top 20 Feature Importances - {target_col}")
        ax.invert_yaxis()
        plt.tight_layout()
        
        logger.report_matplotlib_figure(
            title="Feature Importance",
            series=target_col,
            figure=fig,
            iteration=0
        )
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred_test, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted - {target_col}")
        plt.tight_layout()
        
        logger.report_matplotlib_figure(
            title="Predictions",
            series=target_col,
            figure=fig,
            iteration=0
        )
        plt.close(fig)
    
    return model, test_metrics, importance_df


def train_all_horizons(
    df: pd.DataFrame,
    params: Dict = None,
    task: Task = None
) -> Dict[str, lgb.Booster]:
    models = {}
    all_metrics = {}
    
    for h in range(1, 8):
        target_col = f"target_d{h}"
        
        if target_col not in df.columns:
            print(f"⚠️ Целевая переменная {target_col} не найдена, пропускаем")
            continue
        
        model, metrics, importance = train_model(
            df, 
            target_col=target_col, 
            params=params.copy() if params else None,
            task=task
        )
        
        models[target_col] = model
        all_metrics[target_col] = metrics
    
    if task:
        metrics_df = pd.DataFrame(all_metrics).T
        print("\n📊 Сводка метрик по горизонтам:")
        print(metrics_df.to_string())
        
        task.get_logger().report_table(
            title="Metrics Summary",
            series="All Horizons",
            table_plot=metrics_df
        )
    
    return models


def save_models(
    models: Dict[str, lgb.Booster],
    output_dir: str = None,
    task: Task = None
) -> str:
    if output_dir is None:
        output_dir = MODELS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_paths = {}
    for target_name, model in models.items():
        model_path = output_dir / f"model_{target_name}.txt"
        model.save_model(str(model_path))
        model_paths[target_name] = str(model_path)
        print(f"💾 Сохранена модель: {model_path}")
    
    ensemble_path = output_dir / "weather_ensemble.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(models, f)
    print(f"💾 Сохранен ансамбль моделей: {ensemble_path}")
    
    if task:
        output_model = OutputModel(task=task)
        output_model.update_weights(
            weights_filename=str(ensemble_path),
            auto_delete_file=False
        )
        output_model.update_design(config_dict={
            "model_type": "LightGBM Ensemble",
            "horizons": list(models.keys()),
            "framework": "lightgbm"
        })
        print(f"📤 Модель зарегистрирована в ClearML Model Registry")
    
    return str(ensemble_path)


def main(
    project_name: str = "Weather Forecast",
    task_name: str = "Model Training",
    params: Dict = None,
    register_model: bool = True
):
    print("=" * 60)
    print("🚀 Запуск обучения модели прогнозирования погоды")
    print("=" * 60)
    
    try:
        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            auto_connect_frameworks={"lightgbm": True}
        )
        
        if params:
            task.connect(params)
        
        print(f"📋 ClearML Task ID: {task.id}")
    except Exception as e:
        print(f"⚠️ ClearML не доступен: {e}")
        print("   Продолжаем без логирования в ClearML")
        task = None
    
    df = load_dataset()
    
    if task:
        task.get_logger().report_text(
            f"Dataset info:\n"
            f"- Records: {len(df)}\n"
            f"- Features: {len(df.columns)}\n"
            f"- Date range: {df['date'].min()} to {df['date'].max()}\n"
            f"- Cities: {df['city'].unique().tolist()}"
        )
    
    models = train_all_horizons(df, params=params, task=task)
    model_path = save_models(models, task=task)
    
    if task:
        task.close()
    
    print("\n" + "=" * 60)
    print("✅ Обучение завершено!")
    print(f"📁 Модели сохранены в: {model_path}")
    print("=" * 60)
    
    return models


if __name__ == "__main__":
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50
    }
    
    main(params=default_params)
