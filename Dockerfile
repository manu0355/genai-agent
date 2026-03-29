FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no project install yet)
RUN uv sync --frozen --no-install-project --no-cache

# Copy source
COPY agent/ ./agent/
COPY main.py ./

# Runtime env vars (override via docker run -e or --env-file)
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV XAI_API_KEY=""
ENV DATABASE_URL=""

CMD ["uv", "run", "python", "main.py"]
