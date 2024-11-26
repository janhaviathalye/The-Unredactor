import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier


def load_data(file_path):
    """Load and preprocess the dataset."""
    try:
        # Skip bad lines that don't have the expected number of columns
        data = pd.read_csv(file_path, sep="\t", on_bad_lines='skip', names = ['split', 'name', 'context'])
        print(f"Columns in dataset: {data.columns}")
        # Ensure 'context' is a string before applying regex
        data['cleaned_context'] = data['context'].apply(lambda x: re.sub(r"█+", "REDACTED", str(x)))
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def extract_features(data, vectorizer=None, fit=True):
    """Extract features from the context column using TF-IDF."""
    if fit:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        features = vectorizer.fit_transform(data['cleaned_context'])
    else:
        features = vectorizer.transform(data['cleaned_context'])
    return features, vectorizer


# def train_model(features, labels):
#     """Train a Random Forest model."""
#     X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     return model, X_val, y_val

def train_model(features, labels):
    """Train an ensemble model."""
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model1 = LogisticRegression(max_iter=1000, random_state=42)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    ensemble = VotingClassifier(estimators=[
        ('lr', model1), ('rf', model2)
    ], voting='hard')
    ensemble.fit(X_train, y_train)
    return ensemble, X_val, y_val


def evaluate_model(model, X_val, y_val):
    """Evaluate the trained model."""
    y_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    print(classification_report(y_val, y_pred))
    return precision, recall, f1

def predict_name(model, vectorizer, context):
    """Predict the most likely name for a given redacted context."""
    features = vectorizer.transform([context])
    prediction = model.predict(features)
    return prediction[0]

def generate_submission(model, vectorizer, test_file, output_file):
    """Generate a submission file."""
    try:
        # Skip bad lines in the test file as well
        test_data = pd.read_csv(test_file, sep="\t", on_bad_lines='skip')
        if 'context' not in test_data.columns:
            column_names = ['id', 'context']  # Adjust based on your test data
            test_data = pd.read_csv(test_file, sep="\t", on_bad_lines='skip', names=column_names, header=None)
        test_data['cleaned_context'] = test_data['context'].apply(lambda x: re.sub(r"█+", "REDACTED", str(x)))
        test_data['name'] = test_data['cleaned_context'].apply(lambda x: predict_name(model, vectorizer, x))
        test_data[['id', 'name']].to_csv(output_file, index=False, sep="\t")
        print(f"Submission file saved to {output_file}")
    except Exception as e:
        print(f"Error generating submission: {e}")
        raise


if __name__ == "__main__":
    # Load the data
    data_file = "unredactor.tsv"
    data = load_data(data_file)

    # Filter training and validation data
    train_data = data[data['split'] == 'training']
    val_data = data[data['split'] == 'validation']

    # Extract features
    train_features, vectorizer = extract_features(train_data)
    val_features, _ = extract_features(val_data, vectorizer, fit=False)

    # Train the model
    model, X_val, y_val = train_model(train_features, train_data['name'])

    # Evaluate the model
    precision, recall, f1 = evaluate_model(model, val_features, val_data['name'])
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Generate submission for test data
    test_file = "test.tsv"  # Replace with the actual test file path
    output_file = "submission.tsv"
    generate_submission(model, vectorizer, test_file, output_file)
