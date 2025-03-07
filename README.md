# FAKE-NEWS-PREDICTION-USING-LOGISTIC-REGRESSION

This project implements a Fake News Detection Model using Logistic Regression. The model is trained on a dataset of news articles to classify them as Real (0) or Fake (1) based on their content.

**Dataset:**
The dataset consists of news articles with the following 
**features:**

* Author
* Title
* Text Content
* Label (0: Real News, 1: Fake News)

  **Technologies Used**
* Python
* NumPy, Pandas (Data Handling)
* NLTK (Natural Language Processing - Stopwords, Stemming)
* Scikit-learn (TF-IDF Vectorization, Logistic Regression, Train-Test Split, Accuracy Score)
  **{Why Use Logistic Regression?**
* Effective for Binary Classification – Since the labels are either real or fake, Logistic Regression is well-suited for this problem.
* Interpretable Model – Logistic Regression provides probabilities, making predictions explainable.
* Works Well with TF-IDF – Logistic Regression performs well with textual data transformed into numerical features.
* Efficient for Large Datasets – The model efficiently processes a large number of news articles with text vectorization.**}**
  
**Project Workflow**
* Load the dataset and preprocess missing values.
* Merge the author name and title into a single content column.
* Apply text preprocessing:
* Convert text to lowercase.
* Remove special characters.
* Remove stopwords.
* Apply stemming using Porter Stemmer.
* Convert text into numerical form using TF-IDF Vectorization.
* Split the data into training (80%) and testing (20%) sets.
* Train a Logistic Regression model on the processed dataset.
* Evaluate the model using accuracy score.

**Model Performance**

 **Training Accuracy**
* Accuracy Score: **98.66%**
  **Testing Accuracy**
* Accuracy Score: **97.91%**

**Observations:**
* The model achieves high accuracy in both training and testing, indicating strong generalization.
* The slight drop in test accuracy suggests minimal overfitting.
* Further improvements can be made using hyperparameter tuning or advanced deep learning models like LSTMs or BERT.
  
**Installation & Usage**
Install Dependencies: *pip install numpy pandas nltk scikit-learn*

**Future Improvements**
* Fine-tune hyperparameters for better performance.
* Experiment with advanced models like Random Forest, XGBoost, or Deep Learning models.
* Deploy the model as a web application for real-time fake news detection.

**Author: Arman Ahmad**
[Connect with me on LinkedIn: www.linkedin.com/in/armanahmad16]
