# Machine Learning Final Project
Naive Bayesian Classifier and Bagging

#Overview
This project was developed as the final project for a Machine Learning course.
The objective is to implement a Naive Bayesian Classifier (NBC) from scratch and evaluate the effectiveness of Bagging with different numbers of base models on multiple datasets.
All implementations are written in Python without using machine learning libraries such as scikit-learn.

# Features
- Naive Bayesian Classifier implemented from scratch
- Equal-width discretization (10 bins)
- Laplace smoothing
- Missing value handling
- Bootstrap sampling
- Bagging ensemble (10, 20, 30 base models)
- Majority voting
- 5-fold cross validation
- Statistical evaluation

# Datasets
Three datasets from the UCI Machine Learning Repository were used:
- Banknote Authentication
- Glass Identification
- Image Segmentation

# Statistical Analysis
Performance was evaluated using statistical hypothesis testing based on the course evaluation framework.
The report includes:
- Single dataset aggregation on dataset
- Single dataset aggregation on fold (matched sample)
- Multiple datasets aggregation on dataset
- Multiple datasets aggregation on fold (matched sample)

# Conclusion
Experimental results indicate that Bagging did not consistently outperform a single Naive Bayesian Classifier.
Although Bagging achieved slightly higher estimated accuracy in some datasets, most statistical tests failed to show significant improvements.
Increasing the number of base models also did not consistently improve classification performance.
