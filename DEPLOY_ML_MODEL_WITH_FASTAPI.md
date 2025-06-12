# -Deploying-Vision-API-MachieLearning-Model-Using-FastAPI-Torch-Tensorflow

This guide explains how to deploy a simple trained scikit-learn machine learning model as a web API using FastAPI. FastAPI is a modern, fast web framework for building APIs with Python.

# machine-learning, fastapi, random-forest, model-deployment, artificial-intelligence, python-api, edge-ai


---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Step 1: Prepare Your Trained Model](#step-1-prepare-your-trained-model)
- [Step 2: Create FastAPI App](#step-2-create-fastapi-app)
- [Step 3: Run the FastAPI Server](#step-3-run-the-fastapi-server)
- [Step 4: Test the API](#step-4-test-the-api)
- [Step 5: (Optional) Dockerize the Application](#step-5-optional-dockerize-the-application)
- [Deployment Options](#deployment-options)
- [References](#references)

---

## Introduction

Machine learning models are often deployed as APIs so that they can be consumed by other applications or services. FastAPI is a popular framework for this purpose due to its speed, automatic documentation, and ease of use.

---

## Prerequisites

- Python 3.7+
- Basic knowledge of Python and FastAPI
- A trained scikit-learn model (or willingness to train one)
- `pip` for package installation

---

## Project Structure

Here’s a simple structure for your project:

```
.
├── app/
│   ├── main.py
│   └── model.joblib
├── requirements.txt
└── README.md
```

---

## Step 1: Prepare Your Trained Model

First, train and save your model using scikit-learn and `joblib`. Example:

```python
# train_and_save_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data and train model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "app/model.joblib")
```

Run this script once to save the trained model as `model.joblib`.

---

## Step 2: Create FastAPI App

Create `app/main.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the pre-trained model
model = joblib.load("app/model.joblib")

# Define the data model for requests
class IrisRequest(BaseModel):
    data: list

@app.post("/predict")
def predict(request: IrisRequest):
    # Convert input data to numpy array
    input_data = np.array(request.data)
    # Reshape if single sample
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
```

---

## Step 3: Run the FastAPI Server

Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn joblib numpy
```

Start the server:

```bash
uvicorn app.main:app --reload
```

- The API will be live at `http://127.0.0.1:8000`
- Interactive docs available at `http://127.0.0.1:8000/docs`

---

## Step 4: Test the API

Use `curl`, `httpie`, or Postman to test the endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

Response:

```json
{"prediction": [0]}
```

Or, use the Swagger docs at `/docs` to test with a web interface.

---

## Step 5: (Optional) Dockerize the Application

Create a `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /code
COPY ./app /code/app
COPY requirements.txt /code/

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Add `requirements.txt`:

```
fastapi
uvicorn
scikit-learn
joblib
numpy
```

Build and run:

```bash
docker build -t fastapi-ml-app .
docker run -p 8000:8000 fastapi-ml-app
```

---

## Deployment Options

- **Heroku**: Easy to deploy with Docker or directly from Python.
- **AWS (Elastic Beanstalk, Lambda + API Gateway)**: Scalable options.
- **Azure App Service**: Supports Python web apps.
- **GCP Cloud Run / App Engine**: Good for containerized or Python apps.
- **VPS / Bare metal**: Use Docker or systemd for production.

Refer to the hosting provider’s documentation for deployment steps.

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [joblib Documentation](https://joblib.readthedocs.io/)
- [Deploying FastAPI with Docker](https://fastapi.tiangolo.com/deployment/docker/)
- https://www.youtube.com/watch?v=5PgqzVG9SCk&t=97s

---
