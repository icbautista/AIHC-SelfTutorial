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
    return pd.read_csv(f"{base_path}/{filename}", low_memory=False)

# Define thresholds for each metric for Poor, Fair, Good, Very Good, Excellent
# The thresholds are based on the range [0, 1] for each metric representing the performance in percentage
thresholds = {
    "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],    
    "precision": [0.4, 0.6, 0.7, 0.8, 0.9],
    "recall": [0.4, 0.6, 0.7, 0.8, 0.9],
    "f1_score": [0.4, 0.6, 0.7, 0.8, 0.9]
}
def categorize_performance(score, metric):
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
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Preprocess data function
def preprocess_data(base_path, age_filter=65, filter_greater_than=True, is_training=True, age_range_label="Training Set: Ages > 65"):
    # Load the datasets with the main information
    procedureevents_df = load_csv(base_path, 'PROCEDUREEVENTS_MV.csv')
    admissions_df = load_csv(base_path, 'ADMISSIONS.csv')
    patients_df = load_csv(base_path, 'PATIENTS.csv')

    # Filter the organ support therapies
    organ_support_therapies = procedureevents_df[procedureevents_df['ITEMID'].isin([225792, 225794])]
    print(f"Number of patients before deduplication: {organ_support_therapies['HADM_ID'].nunique()}")
    print(f"Total number of records before deduplication: {organ_support_therapies.shape[0]}")

    # Deduplicate the organ support therapies based on the HADM_ID
    merged_df = organ_support_therapies.merge(admissions_df[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'ADMITTIME']], on='HADM_ID')
    merged_df = merged_df.merge(patients_df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']], on='SUBJECT_ID')

    # Filter the patients based on the age
    merged_df['STARTTIME'] = pd.to_datetime(merged_df['STARTTIME'], errors='coerce')
    merged_df['ENDTIME'] = pd.to_datetime(merged_df['ENDTIME'], errors='coerce')
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'], errors='coerce')
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')
    merged_df.dropna(subset=['STARTTIME', 'ENDTIME', 'ADMITTIME', 'DOB'], inplace=True)

    # Shift the years to have some realistic values for the age
    shift_years = 150
    merged_df['DOB'] = merged_df['DOB'] - pd.DateOffset(years=shift_years)
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    reference_date = today - pd.DateOffset(years=20)
    merged_df = merged_df[(merged_df['DOB'] > pd.to_datetime('1900-01-01')) & (merged_df['DOB'] < reference_date)]
    merged_df['age'] = reference_date.year - merged_df['DOB'].dt.year

    # Filter the patients based on the age for training and evaluation purposes
    if is_training:
        merged_df = merged_df[merged_df['age'] > age_filter]
    else:
        merged_df = merged_df[(merged_df['age'] >= age_filter[0]) & (merged_df['age'] < age_filter[1])]

    merged_df['therapy_duration'] = (merged_df['ENDTIME'] - merged_df['STARTTIME']).dt.total_seconds() / 3600
    merged_df.dropna(subset=['therapy_duration', 'age'], inplace=True)

    # Prepare the features and target
    features = merged_df[['therapy_duration', 'age', 'GENDER']]
    target = merged_df['HOSPITAL_EXPIRE_FLAG']
    features = pd.get_dummies(features, columns=['GENDER'], drop_first=True)
    features = features.apply(pd.to_numeric, errors='coerce')
    features.fillna(0, inplace=True)
    
    # Normalize the features
    scaler = StandardScaler()
    features[['therapy_duration', 'age']] = scaler.fit_transform(features[['therapy_duration', 'age']])

    # Split the data into training and evaluation sets
    smote = SMOTE(random_state=42)
    features_resampled, target_resampled = smote.fit_resample(features, target)
    X_train, X_test, y_train, y_test = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)

    # Display the dataset statistics
    display_dataset_statistics(X_train, X_test, y_train, y_test, age_range_label=age_range_label)
    return X_train, X_test, y_train, y_test

# Display dataset statistics function
def display_dataset_statistics(X_train, X_test, y_train, y_test, age_range_label="Training Set: Ages > 65"):
    num_patients_train = X_train.shape[0]
    num_patients_test = X_test.shape[0]
    avg_age_train = X_train['age'].mean()
    avg_age_test = X_test['age'].mean()
    gender_dist_train = X_train['GENDER_M'].value_counts(normalize=True) * 100
    gender_dist_test = X_test['GENDER_M'].value_counts(normalize=True) * 100
    outcome_dist_train = y_train.value_counts(normalize=True) * 100
    outcome_dist_test = y_test.value_counts(normalize=True) * 100
    avg_therapy_duration_train = X_train['therapy_duration'].mean()
    avg_therapy_duration_test = X_test['therapy_duration'].mean()

    print(f"{age_range_label}: Number of Patients = {num_patients_train}, Average Age = {avg_age_train:.2f}")
    print(f"{age_range_label}: Gender Distribution = {gender_dist_train.to_dict()}")
    print(f"{age_range_label}: Outcome Distribution = {outcome_dist_train.to_dict()}")
    print(f"{age_range_label}: Average Therapy Duration = {avg_therapy_duration_train:.2f} hours")
    print(f"Evaluation Set: Number of Patients = {num_patients_test}, Average Age = {avg_age_test:.2f}")
    print(f"Evaluation Set: Gender Distribution = {gender_dist_test.to_dict()}")
    print(f"Evaluation Set: Outcome Distribution = {outcome_dist_test.to_dict()}")
    print(f"Evaluation Set: Average Therapy Duration = {avg_therapy_duration_test:.2f} hours")

# Function to categorize performance based on thresholds
def categorize_performance(score, metric):
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
    
# Train the model function
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

# Evaluate the model function
def evaluate_model(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

  # Categorize each metric's score
    categories = {
        "accuracy": categorize_performance(accuracy, "accuracy"),
        "precision": categorize_performance(precision, "precision"),
        "recall": categorize_performance(recall, "recall"),
        "f1_score": categorize_performance(f1, "f1_score")
    }

    # Print categorized results
    print(f"Model: {model.__class__.__name__}")
    print(f"Test Accuracy: {accuracy:.4f} ({categories['accuracy']})")
    print(f"Test Precision: {precision:.4f} ({categories['precision']})")
    print(f"Test Recall: {recall:.4f} ({categories['recall']})")
    print(f"Test F1 Score: {f1:.4f} ({categories['f1_score']})")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix

# Plot and save confusion matrix
def save_confusion_matrix_plot(model_name, conf_matrix, age_group_label):
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Summary Plot for {model_name} ({age_group_label})')
    plt.tight_layout(pad=3.0) 
    summary_filename = sanitize_filename(f'shap_summary_plot_{model_name}_{age_group_label}.png')    
    plt.savefig(summary_filename)
    print(f"SHAP summary plot saved as {summary_filename}")
    plt.close()

# Function to save accuracy and F1 performance bar chart
def save_performance_plot(model_name, accuracy, f1, age_group_label):
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy', 'F1 Score'], [accuracy, f1], color=['skyblue', 'salmon'])
    plt.ylim(0, 1)
    plt.title(f'Performance for {model_name} ({age_group_label})')
    filename = sanitize_filename(f'performance_{model_name}_{age_group_label}.png')
    plt.savefig(filename)
    print(f"Performance plot saved as {filename}")
    plt.close()

# Evaluate models on multiple age groups to compare performance across different age ranges
def evaluate_models_on_multiple_age_groups(models, base_path):
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
            train_model(model, X_demo_train, y_demo_train)
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_demo_test, y_demo_test)
            save_confusion_matrix_plot(model_name, conf_matrix, age_group_label)
            save_performance_plot(model_name, accuracy, f1, age_group_label)

            if model_name in ['Random Forest', 'LightGBM-Hyperparameters', 'XGBoost']:
                explain_model_with_shap(model, X_demo_train, X_demo_test, model_name, age_group_label)

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
        train_model(model, X_train, y_train)
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
        save_confusion_matrix_plot(model_name, conf_matrix, "Ages > 65")
        save_performance_plot(model_name, accuracy, f1, "Ages > 65")

        # Explain the model with SHAP for Random Forest, LightGBM, and XGBoost which are tree-based models and work well with SHAP
        # Also, these are the models with the best performance
        if model_name in ['Random Forest', 'LightGBM', 'XGBoost']:
            explain_model_with_shap(model, X_train, X_test, model_name, "Ages > 65")

    # Evaluate models on multiple age groups to compare performance across different age ranges
    evaluate_models_on_multiple_age_groups(all_models, base_path)

# Entry point of the script
if __name__ == "__main__":
    main()
