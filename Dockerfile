# Multi-stage Dockerfile для Weather Forecast ML Pipeline

# ==================== BASE STAGE ====================
FROM python:3.10-slim as base

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# ==================== TRAINING SERVICE ====================
FROM base as training-service

# Устанавливаем зависимости для обучения
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Создаем директории
RUN mkdir -p /app/data /app/models

# Точка входа для обучения
CMD ["python", "-m", "src.train"]

# ==================== INFERENCE SERVICE ====================
FROM base as inference-service

# Устанавливаем зависимости для инференса
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY models/ ./models/

# Создаем директории
RUN mkdir -p /app/models /app/data

# Открываем порт
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск FastAPI сервиса
CMD ["uvicorn", "src.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]

# ==================== DATA LOADER ====================
FROM base as data-loader

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/

# Создаем директории
RUN mkdir -p /app/data

# Точка входа для загрузки данных
CMD ["python", "-m", "src.data_loader"]

# ==================== HPO SERVICE ====================
FROM base as hpo-service

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Создаем директории
RUN mkdir -p /app/data /app/models

# Точка входа для HPO
CMD ["python", "-m", "src.hpo"]

