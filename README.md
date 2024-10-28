# Mortality Risk Prediction for Critical Care Patients

This repository contains code and resources for predicting mortality risk in critically ill patients who are receiving organ support therapies. This project leverages machine learning models to analyze various factors, including age, duration of therapy, and gender, to predict mortality outcomes. The data is based on the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset, a publicly available critical care database.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Features and Models](#features-and-models)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Evaluation and Results] Not included. PDF is shared via CANVAS
---

## Project Overview

This project aims to predict mortality risk among patients in critical care who are undergoing treatments such as mechanical ventilation and vasopressors. The predictions are based on a range of features, which have been selected and preprocessed to improve the accuracy of our models.

## Dataset

The MIMIC-III dataset provides comprehensive de-identified health-related data for over 40,000 critical care patients. To access and use MIMIC-III, you must have a valid PhysioNet account and complete the required training in data handling.

## Features and Models

### Selected Features
- **Therapy Duration**: The length of time the patient is on organ support therapy.
- **Age**: Age of the patient, adjusted for realistic analysis.
- **Gender**: Gender of the patient (male or female).

### Machine Learning Models
The following models were evaluated in this study:
- **Random Forest**
- **LightGBM**
- **XGBoost**
- **Gradient Boosting**
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **AdaBoost**

The models were tuned for performance, and key metrics such as accuracy, precision, recall, and F1 score were calculated to evaluate each model's effectiveness.

## Project Structure

- **data/**: Contains raw data files used for analysis and training. NOT INCLUDED in this project
- **images/** * Stores generated plots, performance metrics, and model output files.
- results.txt * stores log information as part of the process
- **README.md**: Project documentation (this file).

## Installation and Setup

### Requirements
This project requires Python 3.6 or later. The required libraries can be found in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
