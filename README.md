# Churn Modeling

This repository contains code for a churn modeling project. Churn modeling involves predicting whether customers are likely to leave a service or subscription, which is crucial for customer retention and business strategy. The provided code focuses on data preprocessing, exploratory data analysis (EDA), and building a neural network model for churn prediction using Keras.

## Contents

- [Description](#description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Building the Neural Network Model](#building-the-neural-network-model)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Requirements](#requirements)

## Description

The churn modeling code performs the following tasks:

### Data Preprocessing

- Checks the number of unique values in categorical columns.
- Eliminates unnecessary columns like 'RowNumber', 'CustomerId', and 'Surname' for better analysis.
- Handles any null values in the dataset.

### Exploratory Data Analysis (EDA)

- Utilizes visualizations to explore the relationships between categorical variables and the target variable 'Exited'.
- Creates bar plots, countplots, pie charts, box plots, violin plots, and histograms to understand the data distribution and trends.

### Building the Neural Network Model

- Constructs a neural network model using the Keras library.
- Adds layers with varying numbers of units, activation functions, and initializers.
- Compiles the model with the 'binary_crossentropy' loss function.

### Model Evaluation

- Performs model training using the prepared data.
- Evaluates the model using various classification metrics:
  - Confusion Matrix
  - Accuracy Score
  - Precision Score
  - Recall Score
  - F1 Score
  - Classification Report

## Usage

1. Ensure you have the required libraries installed. You can install them using `pip`:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn keras
```

2. Clone this repository:

```bash
git clone https://github.com/your-username/churn-modeling.git
cd churn-modeling
```

3. Download the dataset named 'Churn_Modelling.csv' and place it in the same directory.

4. Run the provided code in your preferred Python environment.

## Requirements

- Python 3.x
- Libraries: pandas, matplotlib, seaborn, numpy, scikit-learn, keras
