# Traffic Flow Prediction in Urban Areas

## Project Overview

This project implements a machine learning approach for predicting traffic congestion levels in an urban area.

The project follows the methodology of the reference paper *Machine Learning for Traffic Flow Prediction in Urban Areas* and adapts it using a newly collected traffic dataset for selected roads around Central Park, New York City.

The system predicts traffic congestion using road information, day of the week, and hour of the day.

---

## Problem Statement

Traffic congestion varies depending on the road, time of day, and day of the week.

The objective of this project is to develop a machine learning model that can classify traffic conditions into five congestion levels:

- Level 0 – Lowest congestion
- Level 1
- Level 2 – Moderate congestion
- Level 3
- Level 4 – Highest congestion

---

## Dataset

Traffic data was collected using Outscraper's Google Maps Traffic Scraper.

The collected data contains information such as:

- Road/route
- Distance
- Travel duration
- Origin coordinates
- Destination coordinates
- Timestamp

The final collected dataset contains 1,071 traffic records.

The data was collected for 35 selected route queries around Central Park, New York City.

---

## Methodology

The project consists of the following stages:

### 1. Data Collection

Traffic information was collected for selected routes around Central Park using historical traffic data.

### 2. Data Preprocessing

The raw traffic data was processed to:

- Convert UTC timestamps to New York local time
- Extract day of the week
- Extract hour of the day
- Calculate average travel speed

Average speed was calculated using:

Speed (km/h) = Distance (km) / Travel Time (hours)

### 3. Congestion Labelling

The calculated speeds were converted into five congestion levels.

Because the newly collected dataset has a different speed distribution from the reference dataset, quantile-based speed grouping was used to obtain five balanced congestion classes.

The final distribution was:

| Congestion Level | Records |
|------------------|---------:|
| 0 | 214 |
| 1 | 214 |
| 2 | 214 |
| 3 | 214 |
| 4 | 215 |

Total records: 1,071

### 4. Baseline Model

A neural network was first trained using:

- Road ID
- Day of week
- Hour

The baseline model achieved:

**39.07% test accuracy**

### 5. Improved Model

The improved model uses:

- One-hot encoding for road
- One-hot encoding for day of week
- Standardized hour feature

The neural network uses:

- Two hidden layers
- 20 neurons in the first hidden layer
- 10 neurons in the second hidden layer
- Logistic activation
- LBFGS solver

The improved model achieved:

**56.74% test accuracy**

This represents an improvement of:

**17.67 percentage points**

over the baseline model.

---

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The project also generates:

- Training loss graph
- Confusion matrix heatmap
- Baseline vs improved accuracy graph

---

## Results

| Model | Test Accuracy |
|-------|--------------:|
| Baseline Neural Network | 39.07% |
| Improved Neural Network | 56.74% |

The improved feature representation resulted in a 17.67 percentage-point increase in test accuracy.

The confusion matrix provides a class-wise view of correct and incorrect predictions across the five congestion levels.

---

## Project Structure

```text
Traffic-Flow-Prediction/
│
├── data/
│   ├── traffic_data.xlsx
│   ├── traffic_processed.csv
│   └── traffic_labelled.csv
│
├── notebooks/
│
├── report/
│
├── results/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   └── traffic_congestion_model.joblib
│
├── src/
│   ├── prepare_data.py
│   ├── create_labels.py
│   ├── train_model.py
│   └── train_improved.py
│
├── venv/
│
└── README.md