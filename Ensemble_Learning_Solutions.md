# Ensemble Learning - Assignment Solutions

## SECTION 1: SKILLS (Theoretical Questions)

### Q1. What is ensemble learning in machine learning?
**Answer:** Ensemble learning is a machine learning technique that combines multiple individual models (base learners) to create a single, more powerful predictive model. The ensemble method leverages the diversity of multiple models to achieve better overall performance than any single model alone. Common ensemble techniques include Bagging, Boosting, and Stacking.

### Q2. What are the main types of ensemble techniques?
**Answer:** 
1. **Bagging (Bootstrap Aggregating)** - Trains multiple models independently on different bootstrap samples and averages predictions
2. **Boosting** - Trains models sequentially, each focusing on mistakes of previous models (AdaBoost, Gradient Boosting, XGBoost)
3. **Stacking** - Uses predictions from multiple base models as input to a meta-model
4. **Voting** - Combines predictions from multiple models using majority voting (classification) or averaging (regression)

### Q3. Explain the key idea behind ensemble techniques
**Answer:** The core idea is the "wisdom of crowds" principle - combining diverse models reduces individual model biases and errors. Even if individual models are weak or make different errors, the ensemble captures different patterns and generalizes better. This diversity is achieved through:
- Different training data (Bagging)
- Different training focus (Boosting)
- Different model architectures (Stacking)
- Different hyperparameters (Voting)

### Q4. What is the main advantage of ensemble techniques?
**Answer:** 
- **Improved Accuracy** - Combines strengths of multiple models
- **Reduced Overfitting** - Averaging reduces variance
- **Reduced Underfitting** - Captures complex patterns better
- **Robustness** - Less sensitive to outliers and noise
- **Better Generalization** - Works well on unseen data
- **Lower variance and bias** - Through averaging and diversity

### Q5. What is the main challenge of ensemble methods?
**Answer:**
- **Computational Complexity** - Training multiple models increases computation time
- **Memory Requirements** - Storing multiple models requires more memory
- **Model Interpretability** - Harder to explain why ensemble made specific predictions
- **Diversity Challenge** - Difficult to ensure sufficient diversity among base models
- **Hyperparameter Tuning** - More hyperparameters to tune
- **Model Selection** - Choosing appropriate base learners

### Q6. When should we avoid using ensemble methods?
**Answer:**
- When a single simple model performs sufficiently well (diminishing returns)
- Real-time prediction systems with strict latency requirements
- Highly constrained memory/computational environments
- When interpretability is critical (e.g., healthcare, finance compliance)
- When training data is very limited
- When individual models are highly correlated (lack of diversity)

### Q7. What is ensemble learning in machine learning? (Explain the concept of feature randomness in Random Forest)
**Answer:** Feature randomness in Random Forest refers to the random selection of features at each node during tree splitting. Instead of searching through all features, Random Forest randomly selects a subset (√n for classification, n/3 for regression). This randomness:
- Increases diversity among trees
- Reduces correlation between trees
- Improves generalization
- Reduces overfitting
- Makes the model more robust to feature scaling

### Q8. What is OOB (Out-of-Bag) Score?
**Answer:** Out-of-Bag (OOB) Score is an internal cross-validation score for Bagging and Random Forest models. Since bootstrap sampling leaves approximately 37% of data out for each tree, this "out-of-bag" data can be used to evaluate model performance without separate validation set:
- Provides unbiased error estimate
- Reduces need for separate validation set
- Computed automatically during training
- Especially useful with limited data
- Calculated as average of predictions on OOB samples

### Q9. How can you measure the importance of features in a Random Forest model?
**Answer:**
1. **Gini/Impurity-based Importance** - Based on how much each feature decreases impurity across all trees
2. **Mean Decrease Impurity (MDI)** - Average decrease in impurity weighted by probability of reaching node
3. **Permutation Importance** - Measures drop in performance when feature values are randomly shuffled
4. **SHAP Values** - Measures feature contribution to individual predictions

Access in scikit-learn: `model.feature_importances_`

### Q10. Explain the working principle of a Bagging Classifier
**Answer:**
1. Create multiple bootstrap samples from training data (sampling with replacement)
2. Train a separate base classifier (e.g., Decision Tree) on each sample
3. Collect predictions from all classifiers
4. For classification: use majority voting
5. For regression: use average of predictions
6. The ensemble prediction is more stable and less prone to overfitting

### Q11. How do you evaluate a Bagging Classifier's performance?
**Answer:**
- **Accuracy** - Percentage of correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of Precision and Recall
- **AUC-ROC** - Area under the ROC curve
- **Cross-Validation** - k-fold cross-validation for robust evaluation
- **Confusion Matrix** - For detailed classification performance
- **OOB Score** - Built-in evaluation metric

### Q12. How does a Bagging Regressor work?
**Answer:**
1. Create multiple bootstrap samples from training data
2. Train a separate base regressor on each sample
3. For regression prediction: average the predictions from all regressors
4. Reduces variance through averaging
5. Benefits from base learners with high variance (e.g., Decision Trees)
6. Works well for complex, non-linear relationships

### Q13. What is the main advantage of ensemble techniques?
**Answer:** (As answered in Q4) - Improved accuracy, reduced overfitting, reduced variance, better generalization, robustness to noise and outliers.

### Q14. What is the main challenge of ensemble methods?
**Answer:** (As answered in Q5) - Computational complexity, memory requirements, interpretability issues, difficulty ensuring diversity.

### Q15. Explain the key idea behind ensemble techniques
**Answer:** (As answered in Q3) - Combining diverse models leverages "wisdom of crowds" to achieve better predictions than individual models.

### Q16. What is a Random Forest Classifier?
**Answer:** Random Forest Classifier is an ensemble learning method that:
- Builds multiple decision trees on bootstrap samples
- Uses random feature selection at each split
- Combines predictions through majority voting
- Works well for classification tasks
- Reduces overfitting through averaging and randomness
- Handles both numerical and categorical data
- Provides feature importance rankings

### Q17. What are the main types of ensemble techniques?
**Answer:** (As answered in Q2) - Bagging, Boosting, Stacking, and Voting.

### Q18. What is ensemble learning in machine learning?
**Answer:** (As answered in Q1) - Combining multiple models for better predictions.

### Q19. When should we avoid using ensemble methods?
**Answer:** (As answered in Q6) - When not needed, real-time systems, memory-constrained, interpretability critical, limited data.

### Q20. How does Bagging help in reducing overfitting?
**Answer:**
- Creates diversity through bootstrap sampling
- Each base model trains on different subset
- Averaging reduces variance from individual model overfitting
- Less prone to fitting noise in training data
- More generalizable to unseen data
- Especially effective with high-variance base learners (Decision Trees)

### Q21. Why is Random Forest better than a single Decision Tree?
**Answer:**
- **Reduced Overfitting** - Ensemble averaging reduces variance
- **Better Generalization** - Multiple perspectives capture patterns better
- **Feature Importance** - More robust feature importance scores
- **Stability** - Less sensitive to small changes in data
- **Handles Non-linearity** - Captures complex relationships
- **Robustness** - Less affected by outliers and noise
- **Reduced Variance** - Through bootstrapping and averaging
- **Lower Bias** - Multiple trees reduce individual biases

### Q22. What is the role of bootstrap sampling in Bagging?
**Answer:**
- Creates different training subsets through sampling with replacement
- Introduces diversity among base models
- Each model sees slightly different data patterns
- Approximately 63.2% unique samples, 36.8% duplicates per bootstrap
- Remaining samples form OOB (Out-of-Bag) data for validation
- Reduces individual model bias without increasing bias
- Key to reducing overfitting through variance reduction

### Q23. Can we use Bagging for regression problems?
**Answer:** **Yes**, absolutely. Bagging is equally applicable to regression problems:
- **Bagging Regressor** averages predictions instead of voting
- **Random Forest Regressor** applies same principles to regression
- Training process identical to classification
- Final prediction is average of all base models
- Works well with high-variance regressors (Decision Trees)
- Reduces prediction variance

### Q24. What is the difference between multiple model training and single model training?
**Answer:**
| Aspect | Multiple Model Training | Single Model Training |
|--------|------------------------|-----------------------|
| Models | Multiple diverse models | One model |
| Training Data | Different subsets/samples | Entire dataset |
| Computation | Higher computational cost | Lower computational cost |
| Overfitting | Reduced through averaging | Higher tendency |
| Accuracy | Generally better | May be lower |
| Interpretability | Complex (multiple models) | Simple (one model) |
| Prediction Time | Slower (predict all models) | Faster (one prediction) |
| Diversity | High (different models) | No diversity |

### Q25. What is the difference between Bagging and Boosting?
**Answer:**
| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel, independent | Sequential |
| **Data Sampling** | With replacement (bootstrap) | Weighted resampling |
| **Error Focus** | All errors equally | Focuses on previous errors |
| **Weight Updates** | No weights | Increases weights on misclassified |
| **Base Learner** | Can be any | Typically weak learners |
| **Variance/Bias** | Reduces Variance | Reduces Bias |
| **Overfitting Risk** | Lower | Higher (sequential focus) |
| **Speed** | Faster (parallel) | Slower (sequential) |
| **Examples** | Random Forest, Bagging | AdaBoost, Gradient Boosting |

### Q26. What are some real-world applications of ensemble techniques?
**Answer:**
1. **Banking & Finance** - Credit risk assessment, fraud detection, loan approval
2. **Healthcare** - Disease prediction, medical diagnosis, treatment recommendations
3. **E-commerce** - Product recommendation, customer churn prediction
4. **NLP** - Sentiment analysis, text classification, machine translation
5. **Computer Vision** - Object detection, image classification, facial recognition
6. **Autonomous Vehicles** - Object detection, trajectory prediction
7. **Weather Prediction** - Combining multiple weather models
8. **Stock Market** - Price prediction, portfolio optimization
9. **Click-through Rate Prediction** - Online advertising
10. **Anomaly Detection** - Detecting unusual patterns in data

---

## SECTION 2: THEORETICAL (Basic Implementations)

### Q1. Train a Bagging Classifier using Decision Trees on a sample dataset and print model accuracy

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create sample dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=10, 
                           n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Bagging Classifier
base_classifier = DecisionTreeClassifier(random_state=42)
bagging_clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)

# Train model
bagging_clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = bagging_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Classifier Accuracy: {accuracy:.4f}")
```

### Q2. Train a Bagging Regressor using Decision Trees and evaluate using Mean Squared Error (MSE)

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Create sample regression dataset
X, y = make_regression(n_samples=100, n_features=20, n_informative=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Bagging Regressor
base_regressor = DecisionTreeRegressor(random_state=42)
bagging_reg = BaggingRegressor(base_estimator=base_regressor, n_estimators=10, random_state=42)

# Train model
bagging_reg.fit(X_train, y_train)

# Predictions and MSE
y_pred = bagging_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Bagging Regressor MSE: {mse:.4f}")
print(f"Bagging Regressor RMSE: {rmse:.4f}")
```

### Q3. Train a Random Forest Classifier on the Breast Cancer dataset and print feature importance scores

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Accuracy
accuracy = rf_clf.score(X_test, y_test)
print(f"Random Forest Classifier Accuracy: {accuracy:.4f}\n")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance Scores:")
print(feature_importance.to_string())
```

### Q4. Train a Random Forest Regressor and compare its performance with a single Decision Tree

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create regression dataset
X, y = make_regression(n_samples=200, n_features=20, n_informative=10, 
                       noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single Decision Tree
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
dt_pred = dt_regressor.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Random Forest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Comparison
print("Decision Tree vs Random Forest:")
print(f"Decision Tree - MSE: {dt_mse:.4f}, R² Score: {dt_r2:.4f}")
print(f"Random Forest - MSE: {rf_mse:.4f}, R² Score: {rf_r2:.4f}")
print(f"\nImprovement: MSE reduced by {((dt_mse - rf_mse)/dt_mse)*100:.2f}%")
```

### Q5. Compute the Out-of-Bag (OOB) Score for a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with OOB score
rf_clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_clf.fit(X_train, y_train)

# OOB Score
oob_score = rf_clf.oob_score_
test_score = rf_clf.score(X_test, y_test)

print(f"Out-of-Bag (OOB) Score: {oob_score:.4f}")
print(f"Test Set Accuracy: {test_score:.4f}")
print(f"\nThe OOB score provides unbiased estimate without need for separate validation set")
```

### Q6. Train a Bagging Classifier using SVM as a base estimator and print accuracy

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging with SVM as base estimator
svm_base = SVC(kernel='rbf', random_state=42)
bagging_svm = BaggingClassifier(base_estimator=svm_base, n_estimators=10, random_state=42)

# Train and evaluate
bagging_svm.fit(X_train, y_train)
y_pred = bagging_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Bagging Classifier with SVM Base Estimator Accuracy: {accuracy:.4f}")
```

### Q7. Train a Random Forest Classifier with different numbers of trees and compare accuracy

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, 
                           n_redundant=5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different numbers of trees
n_estimators_list = [1, 5, 10, 20, 50, 100, 200]
accuracies = []

for n_est in n_estimators_list:
    rf_clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf_clf.fit(X_train, y_train)
    accuracy = rf_clf.score(X_test, y_test)
    accuracies.append(accuracy)
    print(f"n_estimators={n_est:3d} -> Accuracy: {accuracy:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, accuracies, marker='o', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.grid(True)
plt.show()
```

### Q8. Train a Bagging Classifier using Logistic Regression as a base estimator and print AUC score

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging with Logistic Regression
lr_base = LogisticRegression(max_iter=1000, random_state=42)
bagging_lr = BaggingClassifier(base_estimator=lr_base, n_estimators=10, random_state=42)

# Train and evaluate
bagging_lr.fit(X_train, y_train)
y_pred_proba = bagging_lr.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Bagging Classifier with Logistic Regression AUC Score: {auc_score:.4f}")
```

### Q9. Train a Random Forest Regressor and analyze feature importance scores

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Create regression dataset
X, y = make_regression(n_samples=200, n_features=15, n_informative=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'Importance': rf_reg.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance Scores:")
print(feature_importance.to_string())

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Regressor - Feature Importance')
plt.tight_layout()
plt.show()
```

### Q10. Train an ensemble model using both Bagging and Random Forest and compare accuracy

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bagging Classifier
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Comparison
print("Bagging vs Random Forest Comparison:")
print(f"Bagging Accuracy: {bagging_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"\nRandom Forest performs {'better' if rf_acc > bagging_acc else 'worse'} by {abs(rf_acc - bagging_acc):.4f}")
```

---

## SECTION 3: PRACTICAL (Advanced Implementations)

### Q1. Train a Random Forest Classifier and tune hyperparameters using GridSearchCV

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV
rf_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and predictions
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Q2. Train a Bagging Regressor with different numbers of base estimators and compare performance

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Create regression dataset
X, y = make_regression(n_samples=300, n_features=20, n_informative=10, 
                       noise=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different numbers of estimators
n_estimators_list = [1, 5, 10, 20, 50, 100]
results = []

for n_est in n_estimators_list:
    bagging_reg = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=n_est,
        random_state=42
    )
    bagging_reg.fit(X_train, y_train)
    y_pred = bagging_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'n_estimators': n_est, 'MSE': mse, 'R2': r2})
    print(f"n_estimators={n_est:3d} -> MSE: {mse:.4f}, R² Score: {r2:.4f}")

# Convert to DataFrame for visualization
results_df = pd.DataFrame(results)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(results_df['n_estimators'], results_df['MSE'], marker='o', linewidth=2)
ax1.set_xlabel('Number of Estimators')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('MSE vs Number of Estimators')
ax1.grid(True)

ax2.plot(results_df['n_estimators'], results_df['R2'], marker='o', linewidth=2)
ax2.set_xlabel('Number of Estimators')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score vs Number of Estimators')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Q3. Train a Random Forest Classifier and analyze misclassified samples

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)

# Identify misclassified samples
misclassified = y_test != y_pred
misclassified_indices = X_test[misclassified]
misclassified_true = y_test[misclassified]
misclassified_pred = y_pred[misclassified]

# Analyze misclassified samples
print(f"Total Test Samples: {len(y_test)}")
print(f"Correct Predictions: {(y_test == y_pred).sum()}")
print(f"Misclassified Samples: {misclassified.sum()}")
print(f"Accuracy: {(y_test == y_pred).sum() / len(y_test):.4f}\n")

# Display misclassified samples
print("Misclassified Samples Details:")
misclassified_df = pd.DataFrame({
    'True_Label': misclassified_true,
    'Predicted_Label': misclassified_pred,
    'Confidence': [rf_clf.predict_proba(misclassified_indices[i:i+1]).max() 
                   for i in range(len(misclassified_indices))]
})
print(misclassified_df.to_string())
```

### Q4. Train a Bagging Classifier and compare its performance with a single Decision Tree Classifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)

# Bagging Classifier
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)

# Comparison Metrics
print("Decision Tree vs Bagging Classifier Comparison:")
print(f"\n{'Metric':<15} {'Decision Tree':<15} {'Bagging':<15}")
print("-" * 45)

metrics = [
    ('Accuracy', accuracy_score(y_test, dt_pred), accuracy_score(y_test, bagging_pred)),
    ('Precision', precision_score(y_test, dt_pred, average='weighted'), 
     precision_score(y_test, bagging_pred, average='weighted')),
    ('Recall', recall_score(y_test, dt_pred, average='weighted'), 
     recall_score(y_test, bagging_pred, average='weighted')),
    ('F1-Score', f1_score(y_test, dt_pred, average='weighted'), 
     f1_score(y_test, bagging_pred, average='weighted'))
]

for metric_name, dt_val, bagging_val in metrics:
    print(f"{metric_name:<15} {dt_val:<15.4f} {bagging_val:<15.4f}")
```

### Q5. Train a Random Forest Classifier and visualize the confusion matrix

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Random Forest Classifier - Confusion Matrix')
plt.tight_layout()
plt.show()

print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")
```

### Q6. Train a Stacking Classifier using Decision Trees, SVM, and Logistic Regression, and compare accuracy

```python
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(random_state=42)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# Train and evaluate
stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Stacking Classifier Accuracy: {accuracy:.4f}")

# Compare with individual models
print("\nIndividual Base Learner Accuracies:")
for name, model in base_learners:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name.upper()}: {acc:.4f}")
```

### Q7. Train a Random Forest Classifier and print the top 5 most important features

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

# Top 5 features
print("Top 5 Most Important Features:")
print(feature_importance.head(5).to_string(index=False))

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'].head(5), feature_importance['Importance'].head(5))
plt.xlabel('Importance')
plt.title('Top 5 Most Important Features - Random Forest Classifier')
plt.tight_layout()
plt.show()
```

### Q8. Train a Bagging Classifier and evaluate performance using Precision, Recall, and F1-score

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Bagging Classifier
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)
bagging_clf.fit(X_train, y_train)

# Predictions
y_pred = bagging_clf.predict(X_test)

# Evaluate
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Bagging Classifier - Performance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

### Q9. Train a Random Forest Classifier and analyze the effect of max_depth on accuracy

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different max_depth values
max_depth_list = [1, 5, 10, 15, 20, 25, 30, None]
train_accuracies = []
test_accuracies = []

for max_d in max_depth_list:
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=max_d, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    train_acc = rf_clf.score(X_train, y_train)
    test_acc = rf_clf.score(X_test, y_test)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    print(f"max_depth={str(max_d):<5} -> Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# Plot results
depth_labels = [str(d) for d in max_depth_list]
plt.figure(figsize=(10, 6))
plt.plot(depth_labels, train_accuracies, marker='o', label='Train Accuracy', linewidth=2)
plt.plot(depth_labels, test_accuracies, marker='s', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest: Effect of max_depth on Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Q10. Train a Bagging Regressor using different base estimators and compare performance

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Create regression dataset
X, y = make_regression(n_samples=300, n_features=20, n_informative=10, noise=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different base estimators
base_estimators = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
}

results = []

# Train and evaluate with each base estimator
for name, base_est in base_estimators.items():
    bagging_reg = BaggingRegressor(
        base_estimator=base_est,
        n_estimators=10,
        random_state=42
    )
    bagging_reg.fit(X_train, y_train)
    
    y_pred = bagging_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({'Base Estimator': name, 'MSE': mse, 'R² Score': r2})
    print(f"{name}:")
    print(f"  MSE: {mse:.4f}, R² Score: {r2:.4f}\n")

# Compare results
results_df = pd.DataFrame(results)
print("\nComparison Summary:")
print(results_df.to_string(index=False))
```

### Q11. Train a Random Forest Classifier and evaluate its performance using ROC-AUC Score

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Classifier - ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Q12. Train a Bagging Classifier and evaluate its performance using cross-validation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Define Bagging Classifier
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=10,
    random_state=42
)

# Cross-validation with multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted'
}

cv_results = cross_validate(bagging_clf, X, y, cv=5, scoring=scoring)

# Display results
print("Bagging Classifier - 5-Fold Cross-Validation Results:")
print("-" * 60)
print(f"\nAccuracy:")
print(f"  Mean: {cv_results['test_accuracy'].mean():.4f}")
print(f"  Std Dev: {cv_results['test_accuracy'].std():.4f}")
print(f"  Scores: {[f'{s:.4f}' for s in cv_results['test_accuracy']]}")

print(f"\nPrecision (weighted):")
print(f"  Mean: {cv_results['test_precision_weighted'].mean():.4f}")
print(f"  Std Dev: {cv_results['test_precision_weighted'].std():.4f}")

print(f"\nRecall (weighted):")
print(f"  Mean: {cv_results['test_recall_weighted'].mean():.4f}")
print(f"  Std Dev: {cv_results['test_recall_weighted'].std():.4f}")

print(f"\nF1-Score (weighted):")
print(f"  Mean: {cv_results['test_f1_weighted'].mean():.4f}")
print(f"  Std Dev: {cv_results['test_f1_weighted'].std():.4f}")
```

### Q13. Train a Random Forest Classifier and plot the Precision-Recall curve

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

# Calculate Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Classifier - Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Average Precision Score: {avg_precision:.4f}")
```

### Q14. Train a Stacking Classifier with Random Forest and Logistic Regression and compare accuracy

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base learners for stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(random_state=42)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# Train and evaluate
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)

# Compare with individual base learners
print("Model Comparison:")
print("-" * 40)

results = []

# Individual models
for name, model in base_learners:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append({'Model': name.upper(), 'Accuracy': acc})
    print(f"{name.upper()}: {acc:.4f}")

# Stacking model
results.append({'Model': 'STACKING', 'Accuracy': stacking_acc})
print(f"Stacking (RF + DT + LR): {stacking_acc:.4f}")

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))
```

### Q15. Train a Bagging Regressor with different levels of bootstrap samples and compare performance

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Create regression dataset
X, y = make_regression(n_samples=300, n_features=20, n_informative=10, noise=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different bootstrap sample fractions
max_samples_list = [0.2, 0.4, 0.6, 0.8, 1.0]
results = []

for max_samples in max_samples_list:
    bagging_reg = BaggingRegressor(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=10,
        max_samples=max_samples,
        random_state=42
    )
    bagging_reg.fit(X_train, y_train)
    
    y_pred = bagging_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({'max_samples': max_samples, 'MSE': mse, 'R2': r2})
    print(f"max_samples={max_samples} -> MSE: {mse:.4f}, R² Score: {r2:.4f}")

# Visualization
results_df = pd.DataFrame(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(results_df['max_samples'], results_df['MSE'], marker='o', linewidth=2)
ax1.set_xlabel('Bootstrap Sample Fraction')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('MSE vs Bootstrap Sample Fraction')
ax1.grid(True)

ax2.plot(results_df['max_samples'], results_df['R2'], marker='o', linewidth=2)
ax2.set_xlabel('Bootstrap Sample Fraction')
ax2.set_ylabel('R² Score')
ax2.set_title('R² Score vs Bootstrap Sample Fraction')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## Summary

This comprehensive solution covers:

✅ **Skills Section**: 26 theoretical questions with detailed explanations
✅ **Theoretical Section**: 10 practical implementations with code
✅ **Practical Section**: 15 advanced implementations with visualizations

**Key Learnings**:
- Ensemble learning combines multiple models for better performance
- Bagging reduces variance through bootstrap sampling
- Random Forest adds feature randomness for improved diversity
- Stacking uses meta-learners for combining predictions
- Proper hyperparameter tuning is critical for performance
- Evaluation metrics depend on problem type (classification/regression)
- Cross-validation provides robust performance estimates

All code is production-ready and fully commented for easy understanding and implementation.
