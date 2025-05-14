# Spam Email Detection System
#For using the Model contact me at muneerahmed.dev@gmail.com  i will provide the trained model file thank you

This project is a machine learning-based system to detect whether an email is spam or not spam, using natural language processing (NLP) and an artificial neural network (ANN) classifier.

## Overview

The system utilizes Word2Vec embeddings to represent email content and a neural network model to classify messages. The goal is to identify unsolicited or harmful messages (spam) accurately while minimizing false positives.

## Features

- Text preprocessing using spaCy (tokenization, stopword removal, lemmatization)
- Word2Vec embedding for feature representation
- Artificial Neural Network for classification
- High performance on email classification tasks
- Modular and easy to extend

## Technologies Used

- Python
- Word2Vec
- TensorFlow / Keras (ANN)
- spaCy
- Pandas / NumPy

## Dataset

The system was trained on a labeled dataset containing:
- Raw email text
- Binary labels: `spam` or `ham` (not spam)

Datasets such as the SMS Spam Collection or Enron Email Dataset can be used for this purpose.

## Model Pipeline

1. **Preprocessing**
   - Convert to lowercase
   - Remove punctuation and non-alphabetic tokens
   - Tokenize and lemmatize using spaCy
   - Remove stopwords

2. **Feature Extraction**
   - Train or load a pre-trained Word2Vec model
   - Represent each email as an average of its word embeddings

3. **Model Training**
   - Use an ANN with fully connected layers
   - Train on the embedding vectors
   - Evaluate using validation set

4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix

