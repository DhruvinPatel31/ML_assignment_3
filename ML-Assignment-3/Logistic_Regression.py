import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
from autograd import numpy as autonp
import matplotlib.animation as animation

class LogisticRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        return

    #sigmoid function
    def sigmoid(self, y_pred):
        x = autonp.array(y_pred)
        s = (1.0)/ (1 + e**(-x))
        return s

    
    def auto_dmse_unreg(self, theta):
        xt = autonp.dot(np.array(self.X), theta)
        y_pred = self.sigmoid(xt)
        t1 = autonp.dot(self.y.T, autonp.log(y_pred))     #yi*log(y_predi)
        t2 = autonp.ones(self.y.shape) - self.y    #1-yi
        t3 = autonp.log(autonp.ones(self.y.shape) - y_pred)     #log(1-y_predi)
        dmse = -(t1 + autonp.dot(t2, t3))
        return dmse

    def auto_dmse_l1_reg(self, theta):
        xt = autonp.dot(np.array(self.X), theta)
        y_pred = self.sigmoid(xt)
        t1 = autonp.dot(self.y.T, autonp.log(y_pred))     #yi*log(y_predi)
        t2 = autonp.ones(self.y.shape) - self.y    #1-yi
        t3 = autonp.log(autonp.ones(self.y.shape) - y_pred)     #log(1-y_predi)
        dmse = -(t1 + autonp.dot(t2, t3))
        dmse = dmse + (self.lmda * autonp.sum(autonp.abs(theta)))   #adding lambda*|theta| term
        return dmse

    def auto_dmse_l2_reg(self, theta):
        xt = autonp.dot(np.array(self.X), theta)
        y_pred = self.sigmoid(xt)
        t1 = autonp.dot(self.y.T, autonp.log(y_pred))     #yi*log(y_predi)
        t2 = autonp.ones(self.y.shape) - self.y    #1-yi
        t3 = autonp.log(autonp.ones(self.y.shape) - y_pred)     #log(1-y_predi)
        dmse = -(t1 + autonp.dot(t2, t3))
        dmse = dmse + (self.lmda * autonp.dot(theta, theta))    #adding lambda*thetaT*theta term
        return dmse

    def softmax(self, X, theta, k):
        expo = autonp.exp(autonp.dot(X, theta))
        p = expo[:,k] / anp.sum(expo, axis = 1)
        return p

    def auto_dmse_multi(self, theta):
        ext = autonp.exp(autonp.dot(autonp.array(self.X), theta)) 
        ext = ext / autonp.sum(ext, axis = 1)
        dmse = 0
        for i in self.classes:
            dmse = dmse - autonp.dot((self.y == i),autonp.log(ext[:,i]))        
        return dmse


    #unregularized logistic regression
    def fit_non_vectorised(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant'):

        n_samples = len(X)
        self.n_iter = n_iter
        n_batches = n_samples // batch_size
        
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        
        X_batch = [X.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        y_batch = [y.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        theta = np.array([0.0]*X.shape[1]).T
        
        alpha = lr
        for i in range(n_iter):
            if(lr_type == 'inverse'):
                lr = alpha / (i+1)
            
            batch_num = i % n_batches
            X1 = X_batch[batch_num]
            y1 = y_batch[batch_num]
            new_theta = np.array([0.0]*X.shape[1]).T
            for j in range(X.shape[1]):
                dmse = 0
                for k in range(batch_size):
                    y_pred = 0
                    for m in range(X.shape[1]):
                        y_pred = y_pred + (theta[m] * (X1.iloc[k,m]))
                    dmse = dmse + ((self.sigmoid(y_pred) - y1.iloc[k]) * (X1.iloc[k,j]))
                new_theta[j] = theta[j]- lr*dmse/batch_size 
            theta = new_theta    
        self.coef_= theta

        return


    #Unregularized, L1 regularized, L2 regularized using Autograd
    def fit_autograd(self, X, y, batch_size = 5, n_iter=100, lr=0.01, lr_type='constant', lmda = 0.7, reg_type = 'unreg'):

        n_samples = len(X)
        self.n_iter = n_iter
        self.lmda = lmda
        n_batches = n_samples // batch_size
        
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        
        X_batch = [X.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        y_batch = [y.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        theta = np.array([1.0]*X.shape[1]).T
        
        alpha = lr
        for i in range(n_iter):
            if(lr_type == 'inverse'):
                lr = alpha / (i+1)
            
            batch_num = i % n_batches
            X1 = X_batch[batch_num]
            y1 = y_batch[batch_num]
            y_pred = np.dot(X1, theta)
            self.X = X1
            self.y = y1

            #creating gradient function and then calculating gradient
            if(reg_type == "unreg"):
                dmse_grad = grad(self.auto_dmse_unreg)
                dmse = dmse_grad(theta)
            elif(reg_type == "l1_reg"):
                dmse_grad = grad(self.auto_dmse_l1_reg)
                dmse = dmse_grad(theta)
            elif(reg_type == "l2_reg"):
                dmse_grad = grad(self.auto_dmse_l2_reg) 
                dmse = dmse_grad(theta)              

            theta = theta - (lr*dmse)/batch_size

        self.coef_= theta


    def predict(self, X):

        n_samples = len(X)
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        y_pred = np.dot(X, self.coef_)
        y_pred = pd.Series(y_pred)
        y_pred[y_pred < 0] = 0    
        y_pred[y_pred > 0] = 1

        return (y_pred)        

    #Plotting decision boundary
    def plot_decision_boundary(self, X, y):
        fig = plt.figure()
        min_x = -1.5 
        max_x = 1.5
        min_y = -0.5
        max_y = 1.5
        xline = np.array([min_x, max_x])
        c, w1, w2 = list(self.coef_)
        slope = -w1/w2 # Slope
        c = c / (-w2) # Intercept
        yline = slope*xline + c
        plt.plot(xline, yline)
        plt.scatter(X[y==0][0], X[y==0][1], alpha=0.5, s=5, cmap='Paired')
        plt.scatter(X[y==1][0], X[y==1][1], alpha=0.5, s=5,  cmap='Paired')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title("Decision Boundary")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()


    #K-class logistic regressionv (gradient descent)
    def fit_multi_class(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant'):

        n_samples = len(X)
        self.n_iter = n_iter
        n_batches = n_samples // batch_size
        
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        
        X_batch = [X.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        y_batch = [y.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        theta = np.array([0.0]*X.shape[1]).T
        
        cls_list = list(y.unique())
        classes = sorted(cls_list)
        self.classes = classes
        alpha = lr
        for i in range(n_iter):
            if(lr_type == 'inverse'):
                lr = alpha / (i+1)
            
            batch_num = i % n_batches
            X1 = X_batch[batch_num]
            y1 = y_batch[batch_num]
            
            for k in classes:
                cost = -((y1 == k) - self.softmax(X1, theta, k))
                theta[:,k] = theta[:,k] - (lr*np.dot(X1.T, cost))/batch_size
        
        self.coef_ = theta
        return

    #K-class logistic regressionv (Autograd)
    def fit_multi_class_autograd(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant'):

        n_samples = len(X)
        self.n_iter = n_iter
        n_batches = n_samples // batch_size
        
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        
        X_batch = [X.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        y_batch = [y.iloc[i*batch_size : (i+1)*batch_size] for i in range(n_batches)]
        theta = np.array([0.0]*X.shape[1]).T
        
        cls_list = list(y.unique())
        classes = sorted(cls_list)
        self.classes = classes
        alpha = lr
        for i in range(n_iter):
            if(lr_type == 'inverse'):
                lr = alpha / (i+1)
            
            batch_num = i % n_batches
            X1 = X_batch[batch_num]
            y1 = y_batch[batch_num]
            dmse_grad = grad(self.auto_dmse_multi)
            dmse = dmse_grad(theta)
            theta = theta - (lr*dmse)/batch_size
        self.coef_ = theta
        return


    def predict_multi_class(self, X):

        n_samples = len(X)
        if (self.fit_intercept):
            X = pd.DataFrame(np.hstack((np.ones((n_samples,1)), X)))
        xt = np.dot(X, self.coef_)
        y_pred = np.zeros(xt)
        for i in self.classes:
            y_pred[:,k] = self.softmax(X, self.coef_, i)
        maxy = np.argmax(y_pred, axis = 1)
        return maxy