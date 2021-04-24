import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Logistic_Regression import *
from metrics import *
from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(2, size=N))     #integer because y and y_pred must have same class
X = (X - X.min()) / (X.max() - X.min())     #normalized

fit_intercept = True

#L1 type
print("\nRegularized Logistic Regression (L1 type): ")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_autograd(X, y, n_iter = 200, batch_size = 5, lr = 1.5, reg_type = "l1_reg") 
y_hat = LR.predict(X)
print('theta = ', LR.coef_)
print('Accuracy = ', accuracy(y_hat, y))
LR.plot_decision_boundary(X,y)

#L2 type
print("\nRegularized Logistic Regression (L2 type): ")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_autograd(X, y, n_iter = 200, batch_size = 5, lr = 1.5, reg_type = "l2_reg") 
y_hat = LR.predict(X)
print('theta = ', LR.coef_)
print('Accuracy = ', accuracy(y_hat, y))
LR.plot_decision_boundary(X,y)