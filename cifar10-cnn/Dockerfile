FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
