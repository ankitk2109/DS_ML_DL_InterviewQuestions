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
