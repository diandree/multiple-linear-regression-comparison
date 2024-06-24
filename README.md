# Problem Statement 

### 1. Objective
The goal is to develop a predictive model that accurately forecasts the performance index based on various predictors such as hours studied, previous scores, participation in extracurricular activities, sleep hours, and the number os sample question papers practiced.

### 2. Dataset
The dataset consists   of 10,000 students records, each containing the following variables:
1. Hours Studied: The total number of hours spent studying by each student.
2. Previous Scores: The scores obtained by students in previous tests.
3. Extracurricular Activities: Whether the student participates in extracurricular activities. 
4. Sleep Hours: The average number of hours of sleep the student had per day.
5. Sample Question Papers Practiced: The number of sample question papers the student practiced. 

You can find this dataset on [Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression).

The target variable is:
* Performance index: A measure of the overall performance of each student, ranging from 10 to 100, with higher values indicating better performance. 

### 4. Methodology
This problem will be addressed using two different methods:
* Implementing a Multiple Linear Regression model from scratch.
* Implementing a Multiple Linear Regression using the sklearn.linear_model.LinearRegression.class

### Tools
* NumPy: A library for scientific computing, mainly involving linear algebra operations.
* Pandas: A library for data analysis and manipulation.
* Matplotlib: A library for plotting data.
* Scikit-learn (Sklearn): A machine learning library that provides simple and efficient tools for data analysis and machine learning tasks.


### Dataset 

Tasks:
1. Understanding the dataset
2. Extracting data from the dataset into the features and label arrays
3. Splitting data (training and testing sets)

### Splitting Data

We are going to create our train and test sets, with the following distribution from the original dataset:
* 80% of the data will be the training set
* 20% of the data will be the testing set


## 1. Implementing the Multiple Linear Regression from Scratch

### 1.1 Brief explanation

* A Linear Regression is a supervised machine learning algorithm, which means we feed our model with examples that include the right answers.

* Regression predicts a number from infinitely many possible numbers.

* For this dataset, we will implement a Multiple Linear Regression because we have more than one feature in our dataset.


### 1.2 Gradient descent summary
A linear model that predicts $f_{w,b}(x^{(i)})$:
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
In linear regression, we utilize input training data to fit the parameters $w$,$b$ by minimizing a measure of the error between our predictions $f_{w,b}(x^{(i)})$ and the actual data $y^{(i)}$. The measure is called the $cost$, $J(w,b)$. In training you measure the cost over all of our training samples $x^{(i)},y^{(i)}$
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

But for this implementation, we will also analyze the *coefficient of determination*, or $R^2$. There are a number of variants, but the following one is widely used:

$$R^2 = \frac{\text{sum squared regression (SSR)}}{\text{total sum of squares (SST)}} = \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\tag{3}$$ 

 *gradient descent* is described as:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w_{j} &= w_{j} -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{4}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{5}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{6}\\
\end{align}
$$

Here *simultaneously* means that you calculate the partial derivatives for all the parameters before updating any of the parameters.

In order to implement the Multiple Linear Regression from scratch, we need to implement the following functions:
- `compute_cost` implements equation (2)
- `compute_gradient` implements equation (5) and (6)
- `score`: implements equation (3)
- `gradient_descent`, implements equation (4)