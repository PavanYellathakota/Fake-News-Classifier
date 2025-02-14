# naive_bayes_model.py
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class NaiveBayesClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
        ])
        self.model = None
        self.training_time = None
        self.prediction_time = None
        
    def train(self, X_train, y_train):
        """Train the Naive Bayes model"""
        print("Training Naive Bayes model...")
        start_time = time.time()
        
        self.pipeline.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
    def predict(self, X_test):
        """Make predictions using the trained model"""
        start_time = time.time()
        
        predictions = self.pipeline.predict(X_test)
        
        self.prediction_time = time.time() - start_time
        return predictions
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        predictions = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(f"Accuracy score: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print(f"\nTraining time: {self.training_time:.2f} seconds")
        print(f"Prediction time: {self.prediction_time:.2f} seconds")
        
    def predict_single(self, text):
        """Predict for a single text input"""
        prediction = self.pipeline.predict([text])
        return prediction[0]

# Example usage:
if __name__ == "__main__":
    # Import the preprocessor
    from index import DataPreprocessor
    
    # Initialize and prepare data
    preprocessor = DataPreprocessor('FakeNews.csv')
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()
    
    # Train and evaluate model
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train)
    nb_classifier.evaluate(X_test, y_test)
    
    # Example prediction
    sample_text = input("Enter text to predict: ")
    prediction = nb_classifier.predict_single(sample_text)
    print(f"\nPredicted class: {prediction}")
