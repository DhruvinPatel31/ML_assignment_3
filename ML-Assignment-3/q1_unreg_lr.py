import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Logistic_Regression import *
from metrics import *
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(2, size=N))     #integer because y and y_pred must have same class
X = (X - X.min()) / (X.max() - X.min())     #normalized

fit_intercept = True

#Using non-vectorised gradient descent method
print("\nUnregularized Logistic Regression (Gradient Descent): ")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_non_vectorised(X, y, n_iter = 350, batch_size = 5, lr = 2) 
y_hat = LR.predict(X)
print('theta = ', LR.coef_)
print('Accuracy = ', accuracy(y_hat, y))
LR.plot_decision_boundary(X,y)

#Using Autograd
print("\nUnregularized Logistic Regression (Autograd): ")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_autograd(X, y, n_iter = 350, batch_size = 5, lr = 2) 
y_hat = LR.predict(X)
print('theta = ', LR.coef_)
print('Accuracy = ', accuracy(y_hat, y))
LR.plot_decision_boundary(X,y)


#3-Fold 
print("\n3-Fold Unregularized Logistic Regression: ")
X, y = load_breast_cancer(return_X_y=True,as_frame=True)    #load dataset
X = (X - X.min( )) / (X.max( ) - X.min( ))  #normalised
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
K = 3 #no of folds
size = len(data) // K

Xf = [data.iloc[i*size : (i+1)*size].iloc[:,:-1] for i in range(K)]
yf = [data.iloc[i*size : (i+1)*size].iloc[:,-1] for i in range(K)]
avg_accr = 0    #average accuracy over 3 folds
for i in range(K):    
    Xtemp = Xf.copy()
    ytemp = yf.copy()

    Xtest = Xtemp[i]    #keeping 1 fold for test 
    ytest = ytemp[i]

    Xtemp.pop(i)    #remaining for train
    ytemp.pop(i)
    Xtrain = pd.concat(Xtemp)
    ytrain = pd.concat(ytemp)

    LR = LogisticRegression(fit_intercept=True)
    LR.fit_autograd(Xtrain, ytrain, n_iter=100, batch_size = 5, lr = 2)
    y_pred = LR.predict(Xtest)

    accr = accuracy(y_pred, ytest.reset_index(drop=True))
    print("Test Fold-"+ str(i+1))
    print("Accuracy = "+ str(accr))
    avg_accr += accr

avg_accr = avg_accr / K
print("Average Accuracy = " + str(avg_accr))