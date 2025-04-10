# House Price Prediction Model

This project implements a machine learning model to predict house prices based on various features. It includes the complete pipeline from model training to deployment using Flask, Docker, and Google Cloud.

## Project Structure

```
ml-housing-prediction/
├── model/
│   ├── train_model.py       # Script to train and save the model
│   └── model.pkl            # Saved model (generated by train_model.py)
├── app/
│   ├── app.py               # Flask application
│   └── requirements.txt     # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Files to exclude from Docker build
└── README.md                # Project documentation
```

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes, followed by deployment to Google Cloud.

### Prerequisites

- Python 3.9 or higher
- Docker
- Google Cloud account
- Google Cloud SDK (gcloud)

### Local Development

1. **Clone the repository**

```bash
git clone <repository-url>
cd ml-housing-prediction
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r app/requirements.txt
```

4. **Train the model**

```bash
python model/train_model.py
```

This will create a `model.pkl` file in the `model/` directory.

5. **Run the Flask application locally**

```bash
python app/app.py
```

The application will be available at `http://localhost:5000`.

### Testing the API Locally

You can test the API using curl:

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST -H "Content-Type: application/json" -d '{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.023810,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}' http://localhost:5000/predict
```

### Docker Build and Local Testing

1. **Build the Docker image**

```bash
docker build -t housing-prediction:latest .
```

2. **Run the Docker container locally**

```bash
docker run -p 8080:8080 housing-prediction:latest
```

The application will be available at `http://localhost:8080`.

## Deployment to Google Cloud

### Google Cloud Setup

1. **Create a new project** (if needed)

```bash
gcloud projects create [PROJECT_ID] --name="Housing Price Prediction"
gcloud config set project [PROJECT_ID]
```

2. **Enable required APIs**

```bash
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
```

### Deploying the Application

1. **Create a Docker repository in Artifact Registry**

```bash
gcloud artifacts repositories create housing-prediction --repository-format=docker --location=us-central1 --description="Housing prediction model repository"
```

2. **Configure Docker for Google Cloud**

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

3. **Tag the Docker image**

```bash
docker tag housing-prediction:latest us-central1-docker.pkg.dev/[PROJECT_ID]/housing-prediction/app:latest
```

4. **Push the Docker image to Google Cloud**

```bash
docker push us-central1-docker.pkg.dev/[PROJECT_ID]/housing-prediction/app:latest
```

5. **Deploy to Cloud Run**

```bash
gcloud run deploy housing-prediction \
  --image us-central1-docker.
