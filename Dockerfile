# ---- Base Python légère
FROM python:3.11-slim

# Evite les prompts APT
ENV DEBIAN_FRONTEND=noninteractive

# Libs runtime nécessaires (OpenCV a besoin de libgl1 / libglib2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Meilleur cache: installer d'abord les deps Python
COPY requirements.txt .
# (Optionnel mais recommandé) upgrade pip
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code et le dossier data/ (si présent dans le repo)
COPY . .

# L'app lira ici
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

# Render/Heroku injectent $PORT -> on le réutilise tel quel
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

