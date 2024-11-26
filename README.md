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