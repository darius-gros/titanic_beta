# titanic_boosted (beta)


The goal was to predict if the passengers of he Titanic would survive or not: I needed to create an algorithm predicting a binary outcome.
For that challenge, a training dataset with the outcome was provided and we had to train our model on it to predict the outcome of a test set, which of course did not have the outcome in it, that is to say the dependent variable.

Titanic_Kneighbors

The dataset had multiple columns of data, but to begin with, I wanted to create a simple model to see what kind of accuracy I would get from training on just the essential data.
So, on my first model, I used only the 'sex' and passenger class 'Pclass' categories for this classification problem. I trained a couple different models (KNN, Random Forest) on the training set and applied k-fold cross validation to check the mean and variance that the models produced. Accuracy for the best ones seems to converge around 78%. I also tried using a simple ANN algorithm, but it also seemed to converge at a 78% accuracy. I finally choose to use the K Nearest Neighbors classifier because it would give a slightly lower variance than the other classification models. 

EDA XGB Titanic
Trying to improve the model, I did more research on exploratory analysis (EAD), classifier models & how to optimally boost the parameters.
It turned out I needed to do :

1. A better screening of the data

EAD : I started by displaying some graphs using seaborn and matplotlib to have some idea of what the data looked like, what I could do with it. I included more variables such as 'age' & 'embarked' in the classifier. As only 2 values on 418 were missing in the 'embarked' column of the training set, I replace those by the mode. As for the 'age' class, the missing values were taken care of by the xgb classifier. If another classifier had been used, missing age values could have been replaced by the mean of the age given in which class 'Pclass' the person was traveling and not by the global mean, since we saw in EAD, passengers in higher classes were of older age. Then I finished the preprocessing of the data by encoding the categorical data. Feature scaling did not need to be applied thanks to the nature of the XGB Classifier. If a model not tree based had been used such as KMeans, then we would have had to scale the features, with a Standard Scaler for instance.

2. Use XGB Classifier + Use Grid SearchCV to optimize parameters of the XGB Classifier object.

The XGB Classifier produced a higher predictive probability than other classifiers and a plus is that the nature of the XGB Classifier takes care of overfitting on its own thanks to the learning rate parameter. I found that the XGB Classifier combined with thorough parameter tuning via GridSearchCV gave me the highest probability, around 86%. 

Titanic Beta_2

1. Improvement of data cleaning, data visualization and training of the model.
Resulted in a .7% improvement on test set predictions.
