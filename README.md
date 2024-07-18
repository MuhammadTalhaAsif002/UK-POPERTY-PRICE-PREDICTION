
# UK Property Price Prediction Analysis

This repository contains a project that aims to predict property prices in the UK using a dataset from Kaggle. The analysis and model building are performed using PySpark.

## Project Overview

The goal of this project is to predict property prices in the UK using an official dataset. The dataset includes comprehensive information about property transactions. The approach involves data preprocessing, exploratory data analysis (EDA), feature engineering, and the application of machine learning algorithms using PySpark.


## Dataset Description

The dataset contains the following columns:
- Transaction_ID
- price
- Date_of_Transfer
- postcode
- Property_Type
- Old/New
- Duration
- PAON
- SAON
- Street
- Locality
- Town/City
- District
- County
- PPDCategory_Type
- Record_Status - monthly_file_only

## Approach

### 1. Data Preprocessing
- Loaded the dataset using PySpark.
- Handled missing values by either imputing them or dropping the rows/columns with significant missing data.
- Converted categorical variables into numerical formats using techniques such as one-hot encoding.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of property prices.
- Investigated the relationships between different features and the target variable (price).
- Visualized data using histograms, scatter plots, and correlation matrices to identify significant features.

### 3. Feature Engineering
- Extracted useful features from existing columns (e.g., extracting year and month from Date_of_Transfer).
- Created new features that might help improve the prediction model, such as average property price in a postcode area.

### 4. Model Selection and Training
- Split the dataset into training and testing sets.
- Used various regression algorithms to predict property prices, including Linear Regression, Decision Trees, and Random Forest.
- Evaluated the performance of each model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

### 5. Model Evaluation
- Selected the best-performing model based on evaluation metrics.
- Fine-tuned the model using hyperparameter optimization techniques.

### 6. Prediction
- Used the trained model to predict property prices on the test dataset.
- Visualized the predicted vs actual prices to assess the model's performance.

## Results

The project successfully demonstrated the use of PySpark for handling large datasets and building machine learning models. The selected model provided a reasonable prediction accuracy and the approach can be further improved by incorporating more advanced techniques and additional data sources.

## Code Implementation

The code implementation for this project can be found in the `big_data_project.py` script.

## Installation

To run this project, you need to have the following dependencies installed:

- PySpark
- Pandas

You can install these dependencies using pip:

```bash
pip install pyspark pandas
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/uk-property-price-prediction.git
```

2. Navigate to the project directory:

```bash
cd uk-property-price-prediction
```

3. Upload the dataset file (`202304.csv`) to the appropriate directory.

4. Run the script:

```bash
python big_data_project.py
```


## Acknowledgements

- The dataset is provided by Kaggle.
- This project was developed using Google Colab.

## Contact

For any questions or comments, please contact Muhammad Talha Asif at [muhammadtalhaasif90@example.com].
