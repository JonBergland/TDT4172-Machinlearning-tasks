import numpy as np
import pandas as pd

class LogisticRegression():
    
    def __init__(self, epochs = 10000, learning_rate = 0.01):
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.losses, self.train_accuracies = [], []

        pass
        
    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            lin_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid(lin_model)

            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)


            loss = self.compute_loss(y, y_pred) 
            pred_to_class = [1 if _y >= 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class)) 
            self.losses.append(loss)


    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def compute_gradients(self, x, y, y_pred):
        m = x.shape[0]
        error = y_pred - y
        
        grad_w = np.dot(x.T, error) / m
        grad_b = np.mean(error)

        return grad_w, grad_b
        

    def compute_loss(self, y, y_pred):
        m = len(y)
        return - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
    def predict(self, x: pd.DataFrame):
        lin_model = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(lin_model)
        return [1 if _y >= 0.5 else 0 for _y in y_pred]

    def predict_proba(self, x: pd.DataFrame):
        lin_model = np.dot(x, self.weights) + self.bias
        return self.sigmoid(lin_model)