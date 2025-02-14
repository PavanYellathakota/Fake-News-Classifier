# Fake News Detection Using Machine Learning

## Project Overview
This project implements a fake news detection system using machine learning algorithms, specifically Naive Bayes and Support Vector Machine (SVM) classifiers. The system processes text data and classifies news articles as either "FAKE" or "REAL".

## Project Structure
The project is organized into three main Python files, each handling different aspects of the data lifecycle:

### 1. Data Preprocessing (`index.py`)
```python
class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        # ...
```
Key functionalities:
- Data loading and cleaning
- Feature preparation
- Text processing pipeline setup
- Train-test split management

### 2. Naive Bayes Classifier (`naive_bayes_model.py`)
```python
class NaiveBayesClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
```
Features:
- Text vectorization using CountVectorizer
- TF-IDF transformation
- Model training and evaluation
- Single text prediction capability

### 3. SVM Classifier (`svm_model.py`)
```python
class SVMClassifier:
    def __init__(self, kernel='linear'):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(kernel=kernel))
        ])
```
Features:
- Linear kernel SVM implementation
- Text preprocessing pipeline
- Model evaluation metrics
- Prediction functionality

## Performance Metrics
Both models are evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Training and Prediction Time

### Sample Results
#### Naive Bayes Performance:
- Accuracy: ~95.58%
- Training Time: ~0.03 seconds
- Prediction Time: ~0.015 seconds

#### SVM Performance:
- Accuracy: ~93.05%
- Training Time: ~51.82 seconds
- Prediction Time: ~16.56 seconds

## Usage Guide

### 1. Data Preprocessing
```python
from index import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor('path_to_your_dataset.csv')
df = preprocessor.load_data()
X_train, X_test, y_train, y_test = preprocessor.prepare_features()
```

### 2. Training Naive Bayes Model
```python
from naive_bayes_model import NaiveBayesClassifier

nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)
nb_classifier.evaluate(X_test, y_test)
```

### 3. Training SVM Model
```python
from svm_model import SVMClassifier

svm_classifier = SVMClassifier()
svm_classifier.train(X_train, y_train)
svm_classifier.evaluate(X_test, y_test)
```

### 4. Making Predictions
```python
# For single text prediction
text = "Your news article text here"
prediction = classifier.predict_single(text)
```

## Key Features
1. Modular and maintainable code structure
2. Comprehensive evaluation metrics
3. Easy-to-use interface
4. Support for both batch and single-text predictions
5. Performance timing measurements

## Dependencies
- pandas
- scikit-learn
- numpy
- time

## Future Improvements
1. Add support for more classification algorithms
2. Implement cross-validation
3. Add feature importance analysis
4. Implement model persistence
5. Add data visualization components

For more detailed information and presentation materials, please refer to the presentation:
[Fake News Detection Presentation](address_link_of_PPT)

## Project Conclusions
- Both models show strong performance in detecting fake news
- Naive Bayes offers faster training and prediction times
- SVM provides slightly better precision but with longer processing times
- The system demonstrates practical applicability for real-world news classification

---
**Note**: This project is for educational purposes and should be used as part of a broader fact-checking strategy.