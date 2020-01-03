import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
# import seaborn as sns
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline 

import statsmodels.api as sm
import math


# Import models libraries
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(1)

"""
Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
"""



def main():
	# Load Dataset
	dataset = pd.read_csv("Boston.csv")	
	# Get the information of each features
	# print(dataset.info())
	# print(dataset.describe())

	'''
	Detecting outliers
	Visualizing outlier using boxplot and scatter plot
	'''
	# scatter_matrix(dataset)
	# plt.show()

	class DataFrameSelector(BaseEstimator, TransformerMixin):
		def __init__(self, column_name):
			self.column_name = column_name
		def fit(self, X, y = None ):
			return self
		def transform(self, X):
			return X[self.column_name]

	class BackwardElimination():
		def __init__(self, pmax):
			self.pmax = pmax
		def fit(self, X, y = None ):
			return self
		def transform(self, X):
			columns_name = list((dataset.drop('medv', axis = 1)).columns)
			while (len(columns_name)>0):
				p_value = []
				X_new = X[columns_name]
				X_new = sm.add_constant(X_new.to_numpy())
				model = sm.OLS(dataset['medv'],X_new).fit()
				p_value = pd.Series(model.pvalues.values[1:],index = columns_name)      
				self.pmax = max(p_value)
				feature_with_pmax = p_value.idxmax()
				if(self.pmax > 0.05):
					columns_name.remove(feature_with_pmax)
				else:
					break
			columns_name.append('medv')			
			return X[columns_name]

	class OutlierDrop():
		def __init__(self, threshold):
			self.threshold = threshold
		def fit(self, X, y = None ):
			return self
		def transform(self, X):
			print("\n Selected features after Backward Elimination : ", X.columns)
			self.z = np.abs(zscore(X))
			return X[(self.z < threshold).all(axis=1)]

	class SeperateFeaturesLabels():
		def fit(self, X, y = None ):
			return self
		def transform(self, X):
			print("Dataset shape after removing outliers " + str(X.shape) + '\n')
			return X.drop('medv', axis = 1), X['medv'] * 10000

	class TrainTestSplit():
		def __init__(self, size, seed):
			self.size = size
			self.seed = seed
		def fit(self, X, y = None ):
			return self
		def transform(self, X):
			return train_test_split(X[0], X[1], test_size = size, random_state = seed)

	class CrossValidation():
		def __init__(self, models, splits, metric):
			self.models = models
			self.splits = splits
			self.metric = metric
			# self.features = features
			# self.label = label
		def fit(self, X, y = None):
			return self
		def transform(self, X):
			low_std = math.inf
			low_std_model = None
			print('\nModel : Mean, Standard Deviation')
			for model in models:
				model_name = models[model]
				scores = cross_val_score(model_name, X[0], X[2], scoring = metric, cv = splits)
				if scores.std() < low_std:
					low_std = scores.std()
					low_std_model = model
				print('{} : {}, {}'.format(model, round(scores.mean()), round(scores.std())))
			print('\nBest model with low Standard Deviation : ',low_std_model)
			return models[low_std_model]#, X[0], X[2]

	class GridSearchParameters(GridSearchCV):
		def __init__(self, param_grid, cv, metric):
			# self.models = models
			self.param_grid = param_grid
			self.cv = cv
			self.metric = metric
			# self.features = features
			# self.label = label
		def fit(self, X, y = None ):
			return self
		# def predict(self, X, y = None ):
		# 	return self
		def transform(self, X):
			print(metric)
			grid_search = GridSearchCV(X, param_grid, cv, scoring = metric)
			# grid_search.fit(X[1], X[2])			
			return grid_search
			

	'''
	Get the dataset as a numpy array then find the useful feature from it and then remove the outlier.

	Selecting the useful features using Backward elimination method

	Removing the outliers using Standard deviation
	Threshold is given as 3.
	So the z score above 3 are outliers
	'''

	threshold = 3
	pmax = 1

	size = 0.2
	seed = 42

	preprocessing = Pipeline([
		('selector', DataFrameSelector(dataset.columns)),
		('backward_elimination', BackwardElimination(pmax)),
		('outlier', OutlierDrop(threshold)),
		('feature_label', SeperateFeaturesLabels()),
		('train_test_split', TrainTestSplit(size, seed))
		])
	preprocessed_data = preprocessing.fit_transform(dataset)
	X_train, X_test, Y_train, Y_test = preprocessed_data[0], preprocessed_data[1], preprocessed_data[2], preprocessed_data[3]
	
	print("Number of training set : ", X_train.shape)
	print('Number of test set :', X_test.shape)


	models = {}
	models['linear'] = LinearRegression()
	models['SVR'] = SVR(kernel = 'rbf', gamma = 'auto')
	models['DTR'] = DecisionTreeRegressor(max_depth = 3)
	models['RFR'] = RandomForestRegressor(n_estimators=40, random_state = 3)
	models["Lasso"] = Lasso()
	models["ElasticNet"] = ElasticNet()
	models["KNN"] = KNeighborsRegressor()
	models["AdaBoost"] = AdaBoostRegressor(n_estimators=100)
	models["GradientBoost"] = GradientBoostingRegressor()
	models["ExtraTrees"] = ExtraTreesRegressor(n_estimators=100)

	splits =10
	metric = 'neg_mean_squared_error'
	cv = 5
			
	model_search = Pipeline([
		('cross_validation', CrossValidation(models, splits, metric))
		# ('grid_search', GridSearchParameters(param_grid, cv, metric))
		])
	best_model = model_search.fit_transform(preprocessed_data)

	parameters = {
	    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1],
	    "max_depth":[3],
	    "max_features":["log2","sqrt"],
	    "n_estimators":[100]
	    }

	# parameters = {
	# 	'learning_rate' : [0.1],
	# 	'min_samples_split' : [500],
	# 	'min_samples_leaf' : [50],
	# 	'max_depth' : [8],
	# 	'max_features' : ['sqrt'],
	# 	'subsample' : [0.8]
	# 	}

	'''
	Search best combinations of hyperparameter values using GridSearchCV
	'''
	model = GridSearchCV(best_model, parameters, cv = 5, scoring = metric)
	model.fit(X_train, Y_train)
	pred = model.predict(X_test)
	print(model)
	# # print(grid_search.best_estimator_)

	

	# Model Evaluation
	print(" Model Train Accuracy : ", r2_score(Y_train, model.predict(X_train)) * 100)
	print("Model Test Accuracy : ", r2_score(Y_test, model.predict(X_test)) * 100)


	# Plot the predictions
	x_axis = np.array(range(0, pred.shape[0]))
	plt.plot(x_axis, pred, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")
	plt.plot(x_axis, Y_test, linestyle="--", marker="o", alpha=0.7, color='g', label="Y_test")
	plt.title('Predictions vs Y_test')
	plt.legend(loc = 'lower right')
	# plt.savefig("Predictions_vs_Y_test.png")
	plt.show()
	
if __name__ == '__main__':
	main()