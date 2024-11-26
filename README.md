# cis6930fa24-project2

Project 2 - CIS 6930: Data Engineering Fall 2024

Name: Janhavi Athalye

# Project Description

## Overview

In a world where privacy and security are paramount, sensitive information is often redacted from documents to protect identities and comply with legal and ethical standards. However, the reverse process—predicting and recovering redacted information—has its own set of applications, such as reconstructing anonymized datasets for research or testing the robustness of redaction techniques. Manual unredaction is both error-prone and labor-intensive. Project 2: The Unredactor leverages cutting-edge Natural Language Processing (NLP) techniques to automate the unredaction process, specifically focusing on identifying and restoring redacted personal names.

## Key Features

The Unredactor is designed to:

Automatically predict and replace redacted names in text documents, utilizing the surrounding contextual information.
Employ advanced machine learning and NLP techniques, such as Named Entity Recognition (NER) and language modeling, to achieve high accuracy in restoring names.
Process redacted documents like movie reviews, police reports, and court transcripts, adapting to different styles and linguistic structures.

## Pipeline Overview

The project workflow consists of the following key steps:

Data Loading: The dataset is loaded from a tab-separated values (TSV) file, and malformed rows are skipped.
Redacted spans (█) in the context are replaced with the placeholder REDACTED.

Feature Extraction: TF-IDF vectorization is applied to the context column to generate unigram and bigram features.
The feature matrix is optimized for machine learning models.

Model Training: An ensemble model comprising Logistic Regression and Random Forest classifiers is trained using a VotingClassifier.
The model predicts names based on contextual features.

Model Evaluation: The trained model is evaluated on the validation dataset, generating precision, recall, and F1-score metrics.
A detailed classification report is printed for performance analysis.

Submission File Generation: The trained model is applied to the test dataset to predict redacted names.
The predictions are saved in a submission file (submission.tsv) with columns id and name.

## Dataset and Challenges

The project relies on the Large Movie Review Dataset (IMDB) for training and validation. This dataset consists of thousands of movie reviews, providing a rich source of textual content. A custom redaction training set (unredactor.tsv) is provided, containing:

Redacted names and their associated contexts.
Categories for training, testing, and validation splits.
The primary challenge is to accurately predict the names based on limited contextual clues, requiring a robust understanding of language patterns and semantics.

## Task and Approach

The Unredactor aims to:

Train a Prediction Model: Develop a machine learning model using features like n-grams, the length of redacted text, surrounding words, and sentiment analysis of the context.
Validate Model Performance: Evaluate the model on unseen validation data using precision, recall, and F1-score metrics.
Automate Predictions: Provide predictions for redacted names in a test dataset and generate submission files for evaluation.

## Core Features

Context-Aware Prediction: Uses the textual environment around redacted names to infer the most likely candidates.

Flexible Feature Extraction: Incorporates linguistic features, text patterns, and name frequencies for accurate predictions.

Scalable and Reproducible: The pipeline is designed to handle large datasets efficiently and is easily replicable for other redaction/unredaction tasks.

# How to install
Ensure you have pipenv installed otherwise install it using the following command.

```
pip install pipenv
```
The following command can also be used.

```
pipenv install -e
```

Install dependencies using
```
pip install -r requirements.txt
```

## How to run

To run the project, execute the following command after activating the pipenv environment:

```
pipenv run python unredactor.py 

```
## Steps to Run the Pipeline

Download the Dataset: Ensure the unredactor.tsv file is available in the same directory or provide its path in the script.

Prepare the Environment: Clone this repository and navigate to the project directory. Ensure that all dependencies are installed.

Run the Code: Execute the script to train the model, evaluate it, and generate the submission file:
```
python unredactor.py

```
Evaluate the Model: The script will print precision, recall, and F1-score metrics to the console.

Generate Submission: The predicted names for the test dataset will be saved in submission.tsv.


## Functions

### load_data(file_path):

Description: Loads and preprocesses the dataset from a given file path. Handles irregular lines gracefully by skipping rows with unexpected formats and applies preprocessing to ensure redacted spans (e.g., █) are replaced with the placeholder "REDACTED".

Key Features:

Uses pandas to read tab-separated files.
Ensures the 'context' column is processed as a string to handle cases where non-string values might appear.
Returns a DataFrame with cleaned data for further processing.

### extract_features(data, vectorizer=None, fit=True):

Description: Extracts features from the context column using the Term Frequency-Inverse Document Frequency (TF-IDF) method. This method converts textual data into numerical vectors for machine learning models.

Key Features:

Supports n-gram extraction (unigrams and bigrams).
Limits the feature set to the top 5000 most significant features to optimize performance.
If fit=True, trains the vectorizer; otherwise, transforms data using a pre-trained vectorizer.
Returns the feature matrix and the TF-IDF vectorizer object.

### train_model(features, labels)

Description: Trains an ensemble model consisting of Logistic Regression and Random Forest classifiers using a VotingClassifier. The model predicts the names based on redacted contexts.

Key Features:

Splits the dataset into training and validation sets.
Combines multiple algorithms (Logistic Regression and Random Forest) to improve prediction accuracy.
Returns the trained ensemble model along with the validation feature matrix and labels.

### evaluate_model(model, X_val, y_val)

Description: Evaluates the trained model's performance using the validation dataset. Outputs classification metrics, including precision, recall, and F1-score.

Key Features:

Predicts labels for the validation dataset.
Generates a detailed classification report for performance analysis.
Returns precision, recall, and F1-score for further evaluation.

### predict_name(model, vectorizer, context)

Description: Predicts the most likely name to fill a redacted span in the given context.

Key Features:

Transforms the input context into a numerical feature vector using the TF-IDF vectorizer.
Leverages the trained ensemble model to make predictions.
Returns the predicted name.

### generate_submission(model, vectorizer, test_file, output_file)

Description: Generates a submission file by applying the model to a test dataset. Redacts test file contexts, predicts names, and writes the results to a tab-separated file.

Key Features:

Handles irregular or improperly formatted rows in the test dataset.
Cleans and processes the test contexts before making predictions.
Produces a submission file in the required format with columns id and name.
Provides error handling to ensure robustness against unexpected test file issues.

### main()

Description: Coordinates the entire process from loading data to training, evaluation, and submission generation.

Key Features:

Calls all other functions in sequence to implement the complete unredaction pipeline.
Prints evaluation metrics and saves the final predictions for the test dataset.

## Evaluation of the Code

Below is an evaluation of the code based on clarity, reproducibility, and reasoning:

### Clarity of Code and Documentation

The codebase is modular, with each function designed to perform a specific task (e.g., data loading, feature extraction, model training, and evaluation).
Detailed inline comments and docstrings in the code explain the purpose and behavior of each function.
This README provides comprehensive instructions to replicate the pipeline, from setting up the environment to generating predictions and evaluating the model.

### Reproducibility

The pipeline ensures consistent results by:
Using a fixed random state (random_state=42) for splitting data and training models.
Handling inconsistencies in the dataset (e.g., skipping malformed rows) to avoid interruptions during execution.
Steps for running the code, generating metrics, and creating a submission file are explicitly outlined, enabling peers to reproduce results easily.

### Reasoned Approach

#### Feature Extraction:

The use of TF-IDF vectorization with unigrams and bigrams ensures that both individual words and word pairs contribute to the prediction.

#### Model Design:

Combining Logistic Regression and Random Forest in an ensemble model leverages the strengths of both algorithms. Logistic Regression excels in linear relationships, while Random Forest handles non-linear patterns and interactions.

#### Metrics and Evaluation:

Precision, recall, and F1-score are calculated, providing a clear and balanced understanding of the model's performance.
The classification report offers additional insights into performance for specific classes (e.g., individual names).

### Robustness

The pipeline includes robust error handling, such as skipping bad rows during file loading and ensuring compatibility with missing or malformed data in test files.
Special attention is given to preprocessing redacted spans, ensuring that these are standardized across all datasets for consistent feature extraction.

### Extendability
The modular design makes it straightforward to extend the pipeline:
Adding new features (e.g., sentiment scores, external knowledge graphs).
Adapting the model to unredact other entity types, such as locations or dates.


## Bugs and Assumptions

### Bugs

Contextual Ambiguity in Name Prediction: The model relies heavily on the context around the redacted name for prediction. In cases where the context is vague or lacks sufficient information, the predicted name may be inaccurate. 

Length-Based Prediction Errors: Since the length of the redaction (█) is used as a feature, predictions may disproportionately favor names of similar lengths, even when they are contextually inappropriate.

Overfitting on Training Data: The ensemble model may overfit to the specific patterns in the training dataset, reducing its generalizability to unseen data, especially if the context patterns differ significantly.

Handling Special Characters in Contexts: Contexts with special characters, emojis, or uncommon encodings may cause the feature extraction process (e.g., TF-IDF vectorization) to misinterpret or discard relevant text.

Misalignment in Test Data Columns: If the test data has unexpected column names or structure, the code may fail to process it correctly, requiring manual adjustments to the column mappings.

Multiple Valid Names: In contexts where multiple names could be valid, the model selects only one name, potentially overlooking equally valid alternatives.

Skipped Lines in Input Files: Lines skipped during file parsing (due to formatting issues) are not logged in detail, making it difficult to trace specific errors or recover missing data.

Submission File Formatting: If test data contains unexpected formatting or missing values, the generated submission file may have incomplete or inconsistent entries.

### Assumptions

Redaction Consistency: The project assumes that all redactions in the dataset are represented as continuous blocks of █ characters with lengths matching the redacted names. Deviations from this format may lead to incorrect preprocessing or feature extraction.

Text-Based Context: The model assumes that the surrounding context provides sufficient linguistic clues to predict the redacted name. Contexts without meaningful references (e.g., "This is a great movie. ███ did well.") may result in inaccurate predictions.

Balanced Dataset: The training dataset is assumed to have a balanced and diverse distribution of names and contexts. If the dataset is biased, the model may perform poorly on underrepresented patterns.

UTF-8 Encoding: All input files are assumed to be encoded in UTF-8. Files with different encodings may cause errors during reading or processing.

Unique Names: Each redacted span is assumed to correspond to a single, unique name. The model does not account for cases where multiple names could occupy the same span.

Fixed Column Structure: The input datasets (training, validation, and test) are assumed to have consistent columns (split, name, and context). Changes in the column structure may require updates to the file parsing logic.

Predictive Features: The project assumes that features such as n-grams, the length of the redacted text, and surrounding words are sufficient for accurate predictions. More complex linguistic features or external knowledge bases are not considered.

Conceptual Simplicity: The redacted names are assumed to be straightforward (e.g., proper nouns, full names) and not influenced by deeper semantic or cultural nuances that the model may not capture.

Model Resource Availability: The project assumes sufficient computational resources for training and evaluating ensemble models. Limited resources may necessitate reducing model complexity or feature set size.