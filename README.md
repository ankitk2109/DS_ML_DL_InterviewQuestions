# Data Science, Machine Learning, & Deep Learning Interview Questions


## Machine Learning

**1. What is Machine learning?**

Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

Traditionally, software engineering combined human created rules with data to create answers to a problem. Instead, machine learning uses data and answers to discover the rules behind a problem.

To learn the rules governing a phenomenon, machines have to go through a learning process, trying different rules and learning from how well they perform. Hence, why it’s known as Machine Learning.

![image](https://user-images.githubusercontent.com/26432753/90695375-37e31700-e272-11ea-82a4-8a0e7023950e.png)


**2. What is Supervised and Unsupervised Learning?**

Supervised Learning: Uses known and labeled data as input, with feedback mechanism. Most commonly used supervised learning algorithms are decision trees, logistic regression, and support vector machine.

Unsupervised Learning: Uses unlabeled data as input with no feedback mechanism. Most commonly used unsupervised learning algorithms are k-means clustering, hierarchical clustering, and apriori algorithm.

![image](https://user-images.githubusercontent.com/38240162/90569316-955d6200-e1a5-11ea-9a57-c6e2f8ed8fed.png)

![image](https://user-images.githubusercontent.com/38240162/90569357-a8703200-e1a5-11ea-92ec-4f5c71c255cb.png)


**3. What is Logistic Regression?**

Logistic regression measures the relationship between the dependent variable (our label of what we want to predict) and one or more independent variables (our features) by estimating probability using its underlying logistic function (sigmoid).

![image](https://user-images.githubusercontent.com/26432753/90569620-1b79a880-e1a6-11ea-8030-879e3fd2891d.png)

Useful Links: 
  * https://towardsdatascience.com/understanding-logistic-regression-9b02c2aec102
  * https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
  
 
**4. What is a Decision Tree?**

* It is supervised machine learning technique, which can do classification and regression tasks. It is also know as Classification and Regression Trees(CART) algorithm. It formulates the knowledge into hierarchichal structure which are easy to interpret.

* Decision trees are build in two steps.
  * Induction: Process of building the tree
  * Pruning: Removes the unnecessary branches or the branches that do not contribute to the predictive power of classification. It also helps in avoiding overfitting or rigid boundaries.

Useful Links:
  * https://towardsdatascience.com/a-guide-to-decision-trees-for-machine-learning-and-data-science-fe2607241956
  * https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052
  *  How to do regression in Decision tree: https://saedsayad.com/decision_tree_reg.htm#:~:text=The%20ID3%20algorithm%20can%20be,Gain%20with%20Standard%20Deviation%20Reduction.&text=A%20decision%20tree%20is%20built,with%20similar%20values%20(homogenous).
  
  
**5. What is Random forest?**
The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees (which we could call a “forest”), this model uses two key concepts that gives it the name random:
  1. Random sampling of training data points when building trees: The samples are drawn with replacement, known as __bootstrapping__. While testing the model predictions are made by averaging the predictions of each decision tree. This process of random sampling and aggregating the result is know as __bootstrap aggregating__ or __bagging__.
  
  2. Random subsets of features considered when splitting nodes: The other main concept in the random forest is that only a subset of all the features are considered for splitting each node in each decision tree. Generally this is set to sqrt(n_features) for classification meaning that if there are 16 features, at each node in each tree, only 4 random features will be considered for splitting the node.
  
  __The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by averaging the predictions of each individual tree__


Useful links:
  * https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
  * https://towardsdatascience.com/understanding-random-forest-58381e0602d2
  

**6. Bias-Variance tradeoff?**

What is bias?

* Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

What is variance?

* Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.

Bias-Variance?

* If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.
This tradeoff in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more complex and less complex at the same time.


![image](https://user-images.githubusercontent.com/38240162/91093394-473ad980-e651-11ea-90ff-ed392e9de4f8.png)

![image](https://user-images.githubusercontent.com/38240162/91093492-6c2f4c80-e651-11ea-9560-0aea750c3bdd.png)

![image](https://user-images.githubusercontent.com/38240162/91093530-7c472c00-e651-11ea-8c70-98d2ebd3bb85.png)

Useful Links:
 * https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
 * https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
 
**7. What is Naive Bayes Algorithm?**
Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.

![image](https://user-images.githubusercontent.com/26432753/91096656-4789a380-e656-11ea-814d-2edb84ed0ba0.png)

Conclusion:
* Naive Bayes algorithms are mostly used in sentiment analysis, spam filtering, recommendation systems etc. They are fast and easy to implement but their biggest disadvantage is that the requirement of predictors to be independent. In most of the real life cases, the predictors are dependent, this hinders the performance of the classifier.

Useful Links:
 * https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c
 
**8. Handling Imbalanced Data?**

1. Use the right evaluation metrics.
2. Resample the training set: Under Sampling, Over Sampling
3. Use K-fold Cross-Validation in the right way
4. Ensemble different resampled datasets
5. Resample with different ratios

Useful Links:
 * https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
 * https://www.researchgate.net/post/How_10-fold_cross_validation_helps_to_handle_the_imbalance_data_set
 
**9. What is k-fold cross Validation?**
 Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
 Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data
 
 The general procedure is as follows:
  * Shuffle the dataset randomly.
  * Split the dataset into k groups
  * For each unique group:
    - Take the group as a hold out or test data set
    - Take the remaining groups as a training data set
    - Fit a model on the training set and evaluate it on the test set
    - Retain the evaluation score and discard the model
  * Summarize the skill of the model using the sample of model evaluation scores

 Useful Links:
 * https://machinelearningmastery.com/k-fold-cross-validation/
 * https://stats.stackexchange.com/questions/416553/can-k-fold-cross-validation-cause-overfitting
 
**10. What is Ensemble learning? Explain Bagging and Bossting.**

Ensemble Learning: It is the art of combining diverse set of learners (individual models) together to improvise on the stability and predictive power of the model

Bagging and Boosting get N learners by generating additional data in the training stage. N new training data sets are produced by random sampling with replacement from the original set.

Bagging: A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. If samples are drawn with replacement, then the method is known as Bagging.

Boosting: The term 'Boosting' refers to a family of algorithms which converts weak learner to strong learners. Boosting is an ensemble method for improving the model predictions of any given learning algorithm. The idea of boosting is to train weak learners sequentially, each trying to correct its predecessor.

![image](https://user-images.githubusercontent.com/38240162/91494368-a21e3c00-e8b0-11ea-95e1-8e9da5e7c835.png)

Useful Links:
 * https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/
 
**11. Explain Accuracy, Precision, Recall, ROC, F1, Confusion Matrix, RMSE?**

1. Accuracy : the proportion of the total number of predictions that were correct.
2. Positive Predictive Value or Precision : the proportion of positive cases that were correctly identified.
3. Negative Predictive Value : the proportion of negative cases that were correctly identified.
4. Sensitivity or Recall : the proportion of actual positive cases which are correctly identified.
5. Specificity : the proportion of actual negative cases which are correctly identified
6. F1 Score: F1-Score is the harmonic mean of precision and recall values for a classification problem. 
7. RMSE: It is the most popular evaluation metric used in regression problems. It follows an assumption that error are unbiased and follow a normal distribution. As compared to mean absolute error, RMSE gives higher weightage and punishes large errors.
8. ROC: The ROC curve is the plot between sensitivity(TPR) and FPR

Useful Links:
 * https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/
 * https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b#:~:text=We%20can%20use%20classification%20performance,primarily%20used%20by%20search%20engines.
