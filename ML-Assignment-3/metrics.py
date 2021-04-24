import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    y = list(y)
    count = 0
    for i in range(len(y)):
        if(y_hat[i] == y[i]):
            count += 1
    
    return(float(count / len(y)))


def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y_hat = list(y_hat)
    y = list(y)
    tp = 0
    totalp = 0
    for i in range(len(y_hat)):
        if(y_hat[i] == cls):
            if(y_hat[i] == y[i]):
                tp += 1
            totalp += 1
        
    return(float(tp / totalp))

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y_hat = list(y_hat)
    y = list(y)
    tp = 0
    totalt = 0
    for i in range(len(y_hat)):
        if(y[i] == cls):
            if(y_hat[i] == y[i]):
                tp += 1
            totalt += 1
        
    return(float(tp / totalt))

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y_hat = list(y_hat)
    y = list(y)
    mean_sq = 0
    for i in range(len(y_hat)):
        mean_sq += (y_hat[i] - y[i])**2
    
    mean_sq = mean_sq / len(y_hat)
    sqroot = np.sqrt(mean_sq)
    
    return(float(sqroot))

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y_hat = list(y_hat)
    y = list(y)
    abs_err = 0
    for i in range(len(y_hat)):
        abs_err += abs(y_hat[i] - y[i])
    
    mean_err = abs_err / len(y_hat)

    return(float(mean_err))
