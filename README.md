# Travel Insurance Claim Prediction

## 1. Project Overview

This project aims to build a robust machine learning pipeline to predict travel insurance claims, focusing on maximizing recall for the minority class (claim cases) while minimizing business losses due to misclassification. The workflow covers data cleaning, feature engineering, model selection, hyperparameter tuning, evaluation, and business impact analysis.

**Business Objectives**:  
- Accurately identify customers likely to make insurance claims.  
- Minimize financial losses from false negatives (missed claims) and false positives (unnecessary investigations).  
- Provide actionable insights and recommendations for operational and strategic improvements.  

## 2. Data Sources

- [Travel Insurance Data](https://drive.google.com/drive/u/0/folders/1sRZ5xnpsiMID6RFAVPkQN4RLyivdnLAG) : Contains policy, customer, and claim information for travel insurance products.  
  - Features include : `Agency`, `Agency Type`, `Distribution Channel`, `Product Name`, `Destination`, `Duration`, `Net Sales`, `Commission (in value)`, `Age`, and `Claim` status.

## 3. Technologies Used

- **Programming Language** : Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, CatBoost)  
- **Visualization** : Matplotlib, Seaborn  
- **Model Serialization** : Pickle  
- **Notebook Environment** : Jupyter Notebook  
- **Version Control** : Git  

## 4. Project Structure

```
├── README.md          <- Project documentation and workflow summary.
├── data               <- Contains the raw dataset and data dictionary.
│
├── models             <- Trained models and serialized artifacts.
│
├── notebooks          <- Jupyter notebooks for exploration and modeling.
│
├── references         <- Supplementary materials, potentially including detailed data dictionaries or external resource links.
│
├── reports            <- Generated analysis and figures.
│   └── figures        <- Visualizations for reporting.
│
├── requirements.txt   <- Python dependencies for reproducibility.
```

## 5. Workflow Summary

### 5.1 Data Cleaning & Preparation

- **Duplicates** : Removed 4,667 duplicate records (−11.2%).  
- **Missing Values** : Dropped `Gender` due to 71.39% missingness.  
- **Feature Reduction** : Reduced from 11 to 10 features.  
- **Category Grouping** : Consolidated rare categories in `Agency`, `Product Name`, and `Destination` to 'Others' to reduce cardinality and noise.  

### 5.2 Feature Engineering

- **Categorical Encoding** : One-hot encoding for `Agency`, `Agency Type`, `Distribution Channel`, `Product Name`, `Destination`.  
- **Numerical Scaling** : RobustScaler for `Duration`, `Net Sales`, `Commission (in value)`.  
- **Age Binning** : Quantile-based ordinal binning (7 bins) for `Age` to capture non-linear effects and reduce outlier impact.

### 5.3 Model Selection & Benchmarking

- **Imbalance Handling** : Used `scale_pos_weight` and class weighting due to ~98:2 class imbalance.  
- **Model Candidates** : Logistic Regression, SVM, Random Forest, Balanced Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, EasyEnsemble.  
- **Cross-Validation** : Stratified 5-fold, recall as primary metric.  

### 5.4 Hyperparameter Tuning

- **Primary Model** : EasyEnsembleClassifier (with and without AdaBoost base estimator).  
- **Grid Search** : Tuned ensemble size, sampling strategy, AdaBoost parameters.  
- **Threshold Optimization** : Adjusted probability threshold to balance recall and precision.  

### 5.5 Model Evaluation

- **Metrics** : Classification report, confusion matrix, ROC and PR curves.  
- **Business Impact** : Quantified financial losses from FP and FN, demonstrating an estimated 13.27% reduction in annual loss after tuning.  

### 5.6 Model Deployment

- **Serialization** : Final pipeline saved as `Travel_Insurance_Claims_Prediction.sav` using pickle for future inference.  

## 6. Key Results & Insights

- **Recall Improvement** : Minority class recall increased from 0.78 (default) to 0.81 (tuned).  
- **Business Savings** : Annual loss reduced by $4.27M (13.27%) after tuning.  
- **Trade-offs** : Improved recall came with lower precision and accuracy, highlighting the importance of aligning model metrics with business objectives.  

## 7. Recommendations

- **Continuous Monitoring** : Regularly track recall, precision, and business impact metrics.  
- **Data Quality** : Maintain rigorous data cleaning and periodic audits.  
- **Feature Engineering** : Revisit category groupings and explore new features as data evolves.  
- **Model Retraining** : Update models quarterly/biannually to adapt to new trends.  
- **Threshold Review** : Periodically reassess probability thresholds based on business risk tolerance.  
- **Stakeholder Communication** : Provide clear dashboards and documentation for transparency.  
- **Governance** : Version control all scripts, data, and models for reproducibility.  
- **Future Enhancements** : Explore explainability tools (SHAP/LIME), alternative resampling, and integration of external risk factors.  

## 8. How to Use

1. **Environment Setup** :  
     Install dependencies using `pip install -r requirements.txt`.  

2. **Model Inference** :  
     Load the trained model:  
     ```python
     import pickle
     loaded_model = pickle.load(open('Travel_Insurance_Claims_Prediction.sav', 'rb'))
     predictions = loaded_model.predict(X_test)
     ```  

3. **Data Requirements** :  
     Input data must match the feature engineering and preprocessing steps described above.  

## 9. References

- [EasyEnsembleClassifier Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html)  
- [AdaBoostClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [EasyEnsembleClassifier Explanation Diagram](https://i.sstatic.net/W7UmY.png)

## 10. Contact

- Name : Adrian Irshad
- Email : adrianirshad41@gmail.com
- Linkedin : www.linkedin.com/in/adrianirshad
