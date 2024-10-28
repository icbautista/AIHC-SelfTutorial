# Import libraries
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

# Load CSV function
def load_csv(base_path, filename):
    """
    Loads a CSV file from the specified path.

    Args:
        base_path (str): Base directory for the file.
        filename (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Data loaded into a pandas DataFrame.
    """    
    return pd.read_csv(f"{base_path}/{filename}", low_memory=False)

# Define thresholds for each metric for Poor, Fair, Good, Very Good, Excellent
# The thresholds are based on the range [0, 1] for each metric representing the performance in percentage
thresholds = {
    "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],    
    "precision": [0.4, 0.6, 0.7, 0.8, 0.9],
    "recall": [0.4, 0.6, 0.7, 0.8, 0.9],
    "f1_score": [0.4, 0.6, 0.7, 0.8, 0.9]
}

#This function categorizes the performance of the model based on the thresholds above
def categorize_performance(score, metric):
    """
    Categorizes a performance score based on predefined thresholds.

    Args:
        score (float): The metric score.
        metric (str): The type of metric.

    Returns:
        str: The category (Poor, Fair, Good, Very Good, Excellent).
    """    
    if score < thresholds[metric][0]:
        return "Poor"
    elif score < thresholds[metric][1]:
        return "Fair"
    elif score < thresholds[metric][2]:
        return "Good"
    elif score < thresholds[metric][3]:
        return "Very Good"
    else:
        return "Excellent"
    
# Sanitize filename function to remove special characters
def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing special characters with underscores.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: A sanitized filename.
    """    
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Preprocess data function
def preprocess_data(base_path, age_filter=65, filter_greater_than=True, is_training=True, age_range_label="Training Set: Ages > 65"):
    """
    Preprocesses data by loading CSV files, filtering based on age, and applying SMOTE.

    Args:
        base_path (str): Base path for loading files.
        age_filter (int or tuple): Age filter criteria.
        filter_greater_than (bool): Whether to filter ages greater than the age_filter.
        is_training (bool): Whether this is for training data.
        age_range_label (str): Label for age range.

    Returns:
        Tuple containing processed data for training/evaluation: X_train, X_test, y_train, y_test.
    """    
    # Load the datasets with the main information
    procedureevents_df = load_csv(base_path, 'PROCEDUREEVENTS_MV.csv')
    admissions_df = load_csv(base_path, 'ADMISSIONS.csv')
    patients_df = load_csv(base_path, 'PATIENTS.csv')

    # Filter records for organ support therapies using ITEMID codes
    organ_support_therapies = procedureevents_df[procedureevents_df['ITEMID'].isin([225792, 225794])]
    print(f"Number of patients before deduplication: {organ_support_therapies['HADM_ID'].nunique()}")
    print(f"Total number of records before deduplication: {organ_support_therapies.shape[0]}")

    # Merge therapy data with admissions and patient demographics, handling missing date fields    
    merged_df = organ_support_therapies.merge(admissions_df[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'ADMITTIME']], on='HADM_ID')
    merged_df = merged_df.merge(patients_df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']], on='SUBJECT_ID')

    # Filter based on age, calculate therapy duration, and apply data cleaning
    merged_df['STARTTIME'] = pd.to_datetime(merged_df['STARTTIME'], errors='coerce') # Convert to datetime and handle errors
    merged_df['ENDTIME'] = pd.to_datetime(merged_df['ENDTIME'], errors='coerce') 
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'], errors='coerce') 
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce') 
    merged_df.dropna(subset=['STARTTIME', 'ENDTIME', 'ADMITTIME', 'DOB'], inplace=True) # Drop rows with missing values

    # Some adjustments to have more realistic values for the age so we can test different age ranges
    shift_years = 150
    merged_df['DOB'] = merged_df['DOB'] - pd.DateOffset(years=shift_years) # Shift the years
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d')) # Get the current date
    reference_date = today - pd.DateOffset(years=20) # Set the reference date to 20 years ago to filter the patients and have a realistic age
    merged_df = merged_df[(merged_df['DOB'] > pd.to_datetime('1900-01-01')) & (merged_df['DOB'] < reference_date)] # Filter the patients based on the date of birth
    merged_df['age'] = reference_date.year - merged_df['DOB'].dt.year # Calculate the age of the patients

    # Apply age-based filtering for training or evaluation sets
    if is_training: # Filter the patients based on the age for training
        merged_df = merged_df[merged_df['age'] > age_filter]
    else: # Filter the patients based on the age for evaluation
        merged_df = merged_df[(merged_df['age'] >= age_filter[0]) & (merged_df['age'] < age_filter[1])]

    # Calculate therapy duration in hours and clean data
    merged_df['therapy_duration'] = (merged_df['ENDTIME'] - merged_df['STARTTIME']).dt.total_seconds() / 3600
    merged_df.dropna(subset=['therapy_duration', 'age'], inplace=True) # Drop rows with missing values

    # Prepare features (therapy duration, age, gender) and target (hospital mortality)
    features = merged_df[['therapy_duration', 'age', 'GENDER']] # Features
    target = merged_df['HOSPITAL_EXPIRE_FLAG'] # Target which is the hospital expire flag or being dead or alive
    features = pd.get_dummies(features, columns=['GENDER'], drop_first=True) # One-hot encode
    features = features.apply(pd.to_numeric, errors='coerce') # Convert to numeric
    features.fillna(0, inplace=True) # Fill missing values with 0
    
    # Normalize numerical features and apply SMOTE to balance classes
    scaler = StandardScaler() # Initialize the standard scaler used for normalization which means scaling the features to have a mean of 0 and a standard deviation of 1
    features[['therapy_duration', 'age']] = scaler.fit_transform(features[['therapy_duration', 'age']]) # Normalize the features

    # Split the data into training and evaluation sets
    smote = SMOTE(random_state=42) # Initialize the SMOTE which is used for oversampling the minority class to balance the dataset
    features_resampled, target_resampled = smote.fit_resample(features, target) # Resample the features and target
    X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42) # Split the data into training and evaluation sets

    # Display the dataset statistics
    display_dataset_statistics(X_train, X_test, y_train, y_test, age_range_label=age_range_label) # Display the dataset statistics for the training and evaluation sets
    return X_train, X_test, y_train, y_test # Return the training and evaluation sets

# Display statistics for training and evaluation sets
def display_dataset_statistics(X_train, X_test, y_train, y_test, age_range_label="Training Set: Ages > 65"):
    """
    Displays dataset statistics, such as patient count, average age, gender distribution, and outcome distribution.

    Args:
        X_train, X_test: Feature datasets for training and evaluation.
        y_train, y_test: Target datasets for training and evaluation.
        age_range_label (str): Label for the age range.
    """
    num_patients_train = X_train.shape[0] # Get the number of patients in the training set
    num_patients_test = X_test.shape[0] # Get the number of patients in the evaluation set
    avg_age_train = X_train['age'].mean() # Get the average age of the patients in the training set
    avg_age_test = X_test['age'].mean() # Get the average age of the patients in the evaluation set
    
    # GENDER_M is a one-hot encoded column representing the gender of patients.
    # In this encoding, 'GENDER_M' equals 1 if the patient is male, and 0 if female.
    # This approach converts gender into a numerical feature for the model, making it easier to interpret.
    # The 'value_counts(normalize=True) * 100' calculates the percentage of each gender in the dataset,
    # giving us an understanding of gender distribution in both the training and evaluation sets.
    gender_dist_train = X_train['GENDER_M'].value_counts(normalize=True) * 100 
    gender_dist_test = X_test['GENDER_M'].value_counts(normalize=True) * 100
    
    # Get the outcome distribution (percentage of alive vs. deceased patients) in the training and test sets  
    outcome_dist_train = y_train.value_counts(normalize=True) * 100
    outcome_dist_test = y_test.value_counts(normalize=True) * 100
    avg_therapy_duration_train = X_train['therapy_duration'].mean() # Average therapy duration in the training set
    avg_therapy_duration_test = X_test['therapy_duration'].mean() # Average therapy duration in the evaluation set

    # Print dataset statistics
    print(f"{age_range_label}: Number of Patients = {num_patients_train}, Average Age = {avg_age_train:.2f}") # Print the number of patients and average age in the training set
    print(f"{age_range_label}: Gender Distribution = {gender_dist_train.to_dict()}") # Print the demographic distribution in the training set
    print(f"{age_range_label}: Outcome Distribution = {outcome_dist_train.to_dict()}") # Print the outcome distribution in the training set
    print(f"{age_range_label}: Average Therapy Duration = {avg_therapy_duration_train:.2f} hours") # Print the average therapy duration in the training set
    print(f"Evaluation Set: Number of Patients = {num_patients_test}, Average Age = {avg_age_test:.2f}") # Print the number of patients and average age in the evaluation set
    print(f"Evaluation Set: Gender Distribution = {gender_dist_test.to_dict()}") # Print the demographic distribution in the evaluation set
    print(f"Evaluation Set: Outcome Distribution = {outcome_dist_test.to_dict()}") # Print the outcome distribution in the evaluation set
    print(f"Evaluation Set: Average Therapy Duration = {avg_therapy_duration_test:.2f} hours") # Print the average therapy duration in the evaluation set
    
# Train a model using the training dataset 
def train_model(model, X_train, y_train):
    """
    Trains the provided model on the training data.

    Args:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
    """
    model.fit(X_train, y_train)

# Evaluate the model performance and categorize it based on defined thresholds
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance on the test data and categorizes each metric.

    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for test data.

    Returns:
        Tuple of accuracy, precision, recall, f1 score, and confusion matrix.
    """    
    y_pred_test = model.predict(X_test) # Predict the target values for the evaluation set
    accuracy = accuracy_score(y_test, y_pred_test) # Calculate the accuracy of the model
    precision = precision_score(y_test, y_pred_test) # Calculate the precision of the model
    recall = recall_score(y_test, y_pred_test) # Calculate the recall of the model
    f1 = f1_score(y_test, y_pred_test) # Calculate the F1 score of the model
    conf_matrix = confusion_matrix(y_test, y_pred_test) # Calculate the confusion matrix of the model

    # Categorize the performance metrics for readability
    categories = {
        "accuracy": categorize_performance(accuracy, "accuracy"),
        "precision": categorize_performance(precision, "precision"),
        "recall": categorize_performance(recall, "recall"),
        "f1_score": categorize_performance(f1, "f1_score")
    }

    # Print results for easy reference
    print(f"Model: {model.__class__.__name__}")
    print(f"Test Accuracy: {accuracy:.4f} ({categories['accuracy']})")
    print(f"Test Precision: {precision:.4f} ({categories['precision']})")
    print(f"Test Recall: {recall:.4f} ({categories['recall']})")
    print(f"Test F1 Score: {f1:.4f} ({categories['f1_score']})")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix

# Plot confusion matrix for a given model
def save_confusion_matrix_plot(model_name, conf_matrix, age_group_label):
    """
    Saves a plot of the confusion matrix.

    Args:
        model_name (str): Name of the model.
        conf_matrix (np.ndarray): Confusion matrix array.
        age_group_label (str): Label for the age group.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Alive', 'Dead'], yticklabels=['Alive', 'Dead'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name} ({age_group_label})')
    filename = sanitize_filename(f'confusion_matrix_{model_name}_{age_group_label}.png')
    plt.savefig(filename)
    print(f"Confusion matrix plot saved as {filename}")
    plt.close()

# Function to explain the model with SHAP
def explain_model_with_shap(model, X_train, X_test, model_name, age_group_label):
    """
    Generates and saves SHAP summary plots for model interpretability.

    Args:
        model: The trained model.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        model_name (str): Name of the model.
        age_group_label (str): Label for the age group.
    """    
    explainer = shap.TreeExplainer(model) # Initialize the SHAP explainer
    shap_values = explainer.shap_values(X_test) # Calculate SHAP values for the test set
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Use SHAP values for the positive class (deceased patients)

    # SHAP summary plot
    plt.figure(figsize=(10, 6)) # Set the figure size
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False) # Create the SHAP summary plot
    plt.title(f'SHAP Summary Plot for {model_name} ({age_group_label})') # Set the title of the plot
    plt.tight_layout(pad=3.0)  # Adjust the padding
    summary_filename = sanitize_filename(f'shap_summary_plot_{model_name}_{age_group_label}.png')   
    plt.savefig(summary_filename) # Save the SHAP summary plot
    print(f"SHAP summary plot saved as {summary_filename}")
    plt.close()

# Function to save accuracy and F1 performance bar chart
def save_performance_plot(model_name, accuracy, f1, age_group_label):
    """
    Saves a bar plot of Accuracy and F1 Score for a given model.

    Args:
        model_name (str): Name of the model.
        accuracy (float): Accuracy score.
        f1 (float): F1 score.
        age_group_label (str): Label for the age group.
    """    
    plt.figure(figsize=(8, 6)) # Set the figure size
    plt.bar(['Accuracy', 'F1 Score'], [accuracy, f1], color=['skyblue', 'salmon']) # Create the bar plot
    plt.ylim(0, 1) # Set the y-axis limits
    plt.title(f'Performance for {model_name} ({age_group_label})') # Set the title of the plot
    filename = sanitize_filename(f'performance_{model_name}_{age_group_label}.png') # Sanitize the filename
    plt.savefig(filename) # Save the performance plot
    print(f"Performance plot saved as {filename}")
    plt.close()

# Evaluate models on multiple age groups to compare performance across different age ranges
def evaluate_models_on_multiple_age_groups(models, base_path):
    """
    Evaluates models on different age groups and saves relevant performance metrics.

    Args:
        models (dict): Dictionary of models to evaluate.
        base_path (str): Base path for loading files.
    """    
    age_groups = {
        "Ages 0-20": (0, 20),
        "Ages 20-50": (20, 50),
        "Ages 50-65": (50, 65),
        "Ages 0-65": (0, 65)
    }

    for age_group_label, (min_age, max_age) in age_groups.items():
        X_demo_train, X_demo_test, y_demo_train, y_demo_test = preprocess_data(
            base_path,
            age_filter=(min_age, max_age),
            filter_greater_than=False,
            is_training=False,
            age_range_label=f"Evaluation Set: {age_group_label}"
        )

        for model_name, model in models.items():
            train_model(model, X_demo_train, y_demo_train) # Train the model
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_demo_test, y_demo_test) # Evaluate the model
            save_confusion_matrix_plot(model_name, conf_matrix, age_group_label) # Save the confusion matrix plot
            save_performance_plot(model_name, accuracy, f1, age_group_label) # Save the performance plot

            if model_name in ['Random Forest', 'LightGBM-Hyperparameters', 'XGBoost']:
                explain_model_with_shap(model, X_demo_train, X_demo_test, model_name, age_group_label) # Explain the model with SHAP

# Main function to run the entire pipeline / workflow
def main():
    base_path = 'C:\\Users\\Igor Carreon\\Documents\\Other\\Masters\\AI-HC\\MIMICIII'
    X_train, X_test, y_train, y_test = preprocess_data(base_path, age_filter=65, filter_greater_than=True, is_training=True, age_range_label="Training Set: Ages > 65")
    
    all_models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42, learning_rate=0.1, max_depth=-1, n_estimators=300, num_leaves=100, verbosity=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Support Vector Machine': SVC(kernel='linear', class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model in all_models.items():
        train_model(model, X_train, y_train) # Train the model
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test) # Evaluate the model
        save_confusion_matrix_plot(model_name, conf_matrix, "Ages > 65") # Save the confusion matrix plot
        save_performance_plot(model_name, accuracy, f1, "Ages > 65") # Save the performance plot

        # Explain the model with SHAP for Random Forest, LightGBM, and XGBoost which are tree-based models and work well with SHAP
        # Also, these are the models with the best performance
        if model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
            explain_model_with_shap(model, X_train, X_test, model_name, "Ages > 65")

    # Evaluate models on multiple age groups to compare performance across different age ranges
    evaluate_models_on_multiple_age_groups(all_models, base_path)

# Entry point of the script
if __name__ == "__main__":
    main()
