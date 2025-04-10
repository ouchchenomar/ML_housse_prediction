FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app files
COPY model/model.pkl /app/model/
COPY app/app.py .

# Set environment variables
ENV MODEL_PATH=/app/model/model.pkl
ENV PORT=8080

# Expose the port the app will run on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]