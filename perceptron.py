#This is the code for learning how a simple perceptron works, from that published in Raschka's Python Machine Learning textbook, Chapter 2. I do not own this code but have annotated it for the purpose of learning. Please don't copy or distribute as the rights to this code belong to S. Raschka.  

import numpy as np

class Perceptron(object):
	"""Perceptron Classifier.

	Parameters
	------------
	eta : float
		Learning rate (between 0.0 and 1.0)
	n_iter : int 
		Passes over the training dataset

	Attributes
	-----------
	(Convention: underscore is added to atributes that are not being created upon initialisation of the object but but by calling the objects other methods - e.g. self.w_)

	w_ : 1d-array
		Weights after fitting. (w_ is wdelta from text) 
	errors_ : list
		Number of misclassifications in every epoch

	"""
	
	def __init__(self, eta=0.01, n_iter=10): #small learning rate NB
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""Fit training data.

		Parameters
		----------
		X : {array-like}, shape = [n_samples, n_features]
		    Training vectors, where n_samples is the number
		    of samples and n_features is the number of features.
		
		y : array-like, shape = [n_samples]
		    Target values (classifiers?)

		Returns
		-------
		self : object
		
		"""

		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0

		# Aside: zip() will put together N lists containing M elements each, the result having M elements. Each element is a N-tuple.

			for xi, target in zip(X, y):
							# i.e y-yhat from text:
				update = self.eta * (target - self.predict(xi))
				
				self.w_[1:] += update * xi
				self.w_[0] += update
				#i.e if y-yhat != 0 (incorrect classification)
				errors += int(update != 0.0)
				
				self.errors_.append(errors)

		return self
	
	def net_input(self, X):
		
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]
	
	def predict(self, X):
		
		"""Return Class label after unit step"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)









