import numpy as np
import pandas

class LinearRegression():
    
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.epochs = 5000
        self.weights = None
        self.bias = None
        self.learning_rate = 0.001
        self.losses, self.train_accuracies = [], []

        pass
        
    def fit(self, x: pandas.DataFrame, y: pandas.Series):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement

        #Init
        self.weights = np.zeros(x.shape[1])  # Initialize weights based on number of features
        self.bias = 0

        for _ in range(self.epochs):
            # Correct matrix multiplication: x @ weights (not weights @ x.T)
            lin_model = np.dot(x, self.weights) + self.bias
            #print(f"Linmodel: {lin_model}, type: {type(lin_model)}")

            y_pred = lin_model
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)


            loss = self.compute_loss(y, y_pred) 
            # pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] 
            # self.train_accuracies.append(accuracy(y, pred_to_class)) 
            self.losses.append(loss)




        # raise NotImplementedError("The fit method is not implemented yet.")


    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def compute_gradients(self, x, y, y_pred):
        # For linear regression with MSE loss:
        # grad_w = (1/m) * X.T @ (y_pred - y)
        # grad_b = (1/m) * sum(y_pred - y)
        
        m = x.shape[0]  # number of samples
        error = y_pred - y
        
        # Gradient for weights: X.T @ error / m
        grad_w = np.dot(x.T, error) / m
        
        # Gradient for bias: mean of errors
        grad_b = np.mean(error)

        return grad_w, grad_b
        

    def compute_loss(self, y, y_pred):
        # Mean Squared Error for linear regression
        return np.mean((y_pred - y) ** 2)


    
    def predict(self, x: pandas.DataFrame):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        lin_model = np.dot(x, self.weights) + self.bias
        
        return lin_model
        # raise NotImplementedError("The predict method is not implemented yet.")





