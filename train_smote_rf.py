import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    train_precision = precision_score(y_train, train_predictions, average='weighted')
    test_precision = precision_score(y_test, test_predictions, average='weighted')

    train_recall = recall_score(y_train, train_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')

    print("Training Set Metrics:")
    print("Training Accuracy {}: {:.2f}%".format(name, train_accuracy * 100))
    print("Training Precision {}: {:.2f}%".format(name, train_precision * 100))
    print("Training Recall {}: {:.2f}%".format(name, train_recall * 100))

    print("\nTest Set Metrics:")
    print("Test Accuracy {}: {:.2f}%".format(name, test_accuracy * 100))
    print("Test Precision {}: {:.2f}%".format(name, test_precision * 100))
    print("Test Recall {}: {:.2f}%".format(name, test_recall * 100))

def preprocess_data(df):
    scaler = StandardScaler()
    numerical_features = ['longitude', 'latitude', 'Speed_limit', 'hour', 'minute']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def train_and_save_model(data_file, model_file, num_rows=None):
    start_time = time.time()
    print("Loading the dataset...")
    df = pd.read_csv(data_file)

    df[['hour', 'minute']] = df['Time'].str.split(':', expand=True).astype('int32')

    print(f"value counts for {df['Accident_Severity'].value_counts()} ")
    #print(f"columns for {df.columns} ")

    features = ['longitude', 'latitude', 'Speed_limit', 'hour', 'minute','Number_of_Vehicles', 'Number_of_Casualties',
       'Day_of_Week','Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions','2nd_Road_Class','1st_Road_Class','Carriageway_Hazards']
    X = df[features]
    y = df['Accident_Severity']

    pipeline = ImbPipeline([
        ('preprocess', StandardScaler()),
        ('sampling', SMOTE(random_state=20)),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    print("Model Training...")
    # Training the model on the training dataset
    pipeline.fit(X_train, y_train)

    print("Saving the model...")
    # Save the trained model
    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)

    end_time = time.time()
    print(f"Model training and saving took {end_time - start_time:.2f} seconds")
    evaluate_classification(pipeline, "Random Forest", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    data_file = "clean_df.csv"
    model_file = "random_forest_model_smote_train1.pkl"
    num_rows = None  # Set the number of rows for training (e.g., num_rows = 1000000)
    train_and_save_model(data_file, model_file, num_rows)
