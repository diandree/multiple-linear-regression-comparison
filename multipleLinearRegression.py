#Imports
import copy, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------- DATASET --------------------------------------
#1. Understanding the dataset
df = pd.read_csv('Student_Performance.csv')
header = list(df)

print(f'Dataset\n {df}')

#Verifying if there are null values in the dataset
print(f'Number of null data:\n{df.isnull().sum()}')

#Data description
print(f'\n{df.describe()}')

#Replacing Yes/No features with 0 or 1
df = df.replace({'Yes':1, 'No':0}).infer_objects(copy=False)


#Splitting variables 
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_set = X.to_numpy()
y_set = y.to_numpy()

print(f"X set shape: {X_set.shape}\ny set shape: {y_set.shape}")

fig,ax=plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_set[:,i],y_set)
    ax[i].set_xlabel(header[i])
ax[0].set_ylabel("Performance index")
plt.show()

fig,ax=plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].hist(X_set[:,i])
    ax[i].set_xlabel(header[i])
ax[0].set_ylabel("Count")
plt.show()
#------------------------------------------------------------------------------

# --------------------- DATA NORMALIZATION -------------------------------- 
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)


# normalize the original features
X_set_norm, X_mu, X_sigma = zscore_normalize_features(X_set)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")

fig,ax=plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_set_norm[:,i],y_set)
    ax[i].set_xlabel(header[i])
ax[0].set_ylabel("Performance index")
plt.show()
#------------------------------------------------------------------------------

# ------------------------------- DATA SPLITTING ----------------------------------
#Splitting data
total_samples = X_set_norm.shape[0]
# Test data percentage - 20%
test_percentage = 0.20

# Number of test samples
n_test_samples = int(total_samples * test_percentage)

# Generate random indices
indices = np.arange(total_samples)
np.random.shuffle(indices)

# Split indices into training and test sets
test_indices = indices[:n_test_samples]
train_indices = np.setdiff1d(indices, test_indices)

X_train, X_test = X_set_norm[train_indices], X_set_norm[test_indices]
y_train, y_test = y_set[train_indices], y_set[test_indices]

print(f"X_train shape: {X_train.shape}\tX_test shape: {X_test.shape}\ny_train shape: {y_train.shape}\t\ty_test shape: {y_test.shape}")

fig,ax=plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].hist(X_train[:,i])
    ax[i].set_xlabel(header[i])
ax[0].set_ylabel("Count")
plt.show()
#------------------------------------------------------------------------------

# ----------------------------- MULTIPLE LINEAR REGRESSION --------------------------------
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + ((f_wb_i - y[i])**2)
    cost = cost/(2*m)
    return cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        err = f_wb_i - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iter):
    w = copy.deepcopy(w_in) #avoid modifying global w within function
    b = b_in
    J_history = []

    for i in range(num_iter):
        #Calculate gradients
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        #Update parameters w and b
        w -= (alpha * dj_dw)
        b -= (alpha * dj_db)

        #Calculate J 
        cost = compute_cost(X, y, w, b)
        if (len(J_history) > 1) and (J_history[-1] - cost) < 1e-10:
            print(cost)
            break 
        
        J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
        
    #Return final w,b and J history for graphing    
    return w, b, J_history 
#------------------------------------------------------------------------------

# ----------------------- RUNNING MULTIPLE LINEAR REGRESSION ---------------------------
#Initializing data
initial_w = np.random.rand(X_train.shape[1])
initial_b = np.random.randn()

alpha = 0.001
num_iter = 50000

#Running gradient descent - with normalized data
final_w, final_b, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, num_iter)

print(f"For learning rate = {alpha} ---- b, w found by gradient descent: {final_b:0.2f}, {final_w}")
#------------------------------------------------------------------------------

# ------------------------------- EVALUATION -------------------------------------------
def r2_score(y, preds):
    v = np.sum((y - np.mean(y)) ** 2) # Calculate the total sum of squares
    u = np.sum((y - preds) ** 2) # Calculate the residual sum of squares (RSS)
    r2 = 1 - (u / v) # Calculate the R^2 score

    return r2

def prediction(X, final_w, final_b):
    m = X.shape[0]
    preds = []

    for i in range(m):
        f_wb = np.dot(X[i], final_w) + final_b
        preds.append(f_wb)
    return preds

# Calculating predictions
preds = prediction(X_test, final_w, final_b)
r2 = r2_score(y_test, preds)
mse = compute_cost(X_test, y_test, final_w, final_b)

print(f'R^2: {r2:.4f}')
print(f'MSE: {mse:.4f}')

#Plot cost versus iteration  
plt.plot(J_hist, '-')
plt.xlabel('Iterations step')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

#Plot predictions vs targets  
fig,ax=plt.subplots(1,5,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_test[:,i],y_test, label = 'target')
    ax[i].set_xlabel(header[i])
    ax[i].scatter(X_test[:,i],preds, label = 'predict')
ax[0].set_ylabel("Performance Index"); ax[0].legend();
fig.suptitle("Target vs. Prediction (z-score normalized model)")
plt.show()


# ----------------------- MULTIPLE LINEAR REGRESSION SKLEARN IMPLEMENTATION --------------------------
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Defining the model
regressor = LinearRegression().fit(X_train, y_train)
print(f'\nR2 score (training set): {regressor.score(X_train, y_train):.4f}')

#Making predictions
y_pred_test = regressor.predict(X_test)
sk_r2_score = regressor.score(X_test, y_test)
sk_mse = mean_squared_error(y_test, y_pred_test)

print(f'R2 score (testing set): {sk_r2_score:.4f}')
print(f'\nMSE: {sk_mse}')
print(f'\nCoefficient (w): {regressor.coef_}')
print(f'\nIntercept (b): {regressor.intercept_:.2f}')

print(X_test.shape, y_test.shape)

# Prediction on testing set
fig,ax=plt.subplots(1,5,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_test[:,i],y_test, label = 'target')
    ax[i].set_xlabel(header[i])
    ax[i].scatter(X_test[:,i], y_pred_test, label = 'predict')
ax[0].set_ylabel("Performance Index"); ax[0].legend();
fig.suptitle("Target vs. Prediction (Sklearn Model)")
plt.show()
#------------------------------------------------------------------------------

# -------------------------- COMPARISON -------------------------------
print(f'MSE:\tSimple LR={J_hist[-1]:0.2f}\t\t\tSklearn={sk_mse:.2f}')
print(f'R2:\tSimple LR={r2:0.4f}\t\tSklearn={sk_r2_score:.4f}')
print(f'b:\tSimple LR={final_b:0.2f}\t\t\tSklearn={regressor.intercept_:.2f}')
print(f'w:\tSimple LR={final_w}\nw: Sklearn={regressor.coef_}')