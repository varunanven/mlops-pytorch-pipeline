# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline designed to train, package, and deploy PyTorch image classification models. This repository features data ingestion, automated containerised workloads for training and inference, and robust infrastructure blueprints for production Kubernetes clusters.

---

## 📦 Repository Structure

```text
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow for unit tests and linting
├── configs/
│   └── training_config.yaml     # Hyperparameters, data paths, and infrastructure settings
├── docker/
│   ├── Dockerfile.train         # Container blueprint for model training environments
│   └── Dockerfile.serve         # Container blueprint for FastAPI deployment
├── k8s/
│   ├── namespace.yaml           # Isolated cluster workspace definition
│   ├── configmap.yaml           # Environment variables and configuration tracking
│   ├── training-job.yaml        # Short-lived Kubernetes Job for fault-tolerant training
│   ├── serving-deployment.yaml  # Scalable stateless pod replicas for inference hosting
│   ├── serving-service.yaml     # Internal load balancer mapping traffic to endpoints
│   └── hpa.yaml                 # Horizontal Pod Autoscaler tracking CPU/custom metrics
├── requirements/
│   ├── train.txt                # Training dependencies (PyTorch, TorchVision, YAML parsers)
│   └── serve.txt                # Inference dependencies (FastAPI, Uvicorn, Python-Multipart)
├── src/
│   ├── dataset.py               # Custom PyTorch Dataset/DataLoader definitions
│   ├── model.py                 # Core Neural Network architecture graph definition
│   ├── train.py                 # Distributed/Single-GPU model optimization loop
│   └── serve.py                 # FastAPI application layer for real-time image inference
└── tests/
    └── test_model.py            # Diagnostic unit tests verifying tensor input/output shapes
```

---

##  Getting Started

Ensure you have Docker and Python installed locally before proceeding.

### Containerised Training & Inference

We build isolated, lightweight target artifacts to keep runtime dependencies separated for training and serving tasks.

#### 1. Model Training Cycle
Build the training environment and run the pipeline by mounting your local data directories and checkpoint outputs to persist the model state:

```bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run training with mounted volumes
docker run --rm \
  -v \$(pwd)/data:/app/data \
  -v \$(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

#### 2. Model Serving & Inference Deployment
Build the lightweight application runtime environment. Mount your checkpoint directory containing the trained PyTorch weights file into the inference runtime:

```bash
# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving
docker run --rm -p 8080:8080 \
  -v \$(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

#### 3. Test Prediction Endpoint
Verify that your inference server is responding accurately by passing a test image payload through HTTP POST:

```bash
# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```
