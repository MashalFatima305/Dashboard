#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Updated data for confusion matrices
confusion_matrices = {
    'OLS': [
        [-1.0, 0.0, 1.0, 2.0, 3.0, 197142903.0],
        [-1.0, 0, 0, 0, 0, 0, 0],
        [0.0, 5, 1137, 170, 1, 0, 0],
        [1.0, 1, 10, 36, 2, 1, 1],
        [2.0, 0, 0, 0, 0, 0, 0],
        [3.0, 0, 0, 0, 0, 0, 0],
        [197142903.0, 0, 0, 0, 0, 0, 0]
    ],
    'Ridge': [
        [0, 0, 0, 0, 0, 0],
        [5, 1129, 179, 0, 0, 0],
        [0, 7, 41, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    'Lasso': [
        [826, 487, 0],
        [20, 30, 1],
        [0, 0, 0]
    ],
    'RFE Logistic Regression': [
        [726, 4],
        [587, 47]
    ],
    'Polynomial Logistic Regression': [
        [1106, 207],
        [30, 21]
    ]
}

# Example data for bar graphs
metrics = {
    'Model': ['Ridge', 'Lasso', 'ElasticNet', 'Polynomial Logistic'],
    'Accuracy': [0.85, 0.80, 0.82, 0.88],
    'Recall': [0.80, 0.75, 0.78, 0.85],
    'Precision': [0.83, 0.78, 0.80, 0.87],
    'F1 Score': [0.82, 0.76, 0.79, 0.86]
}
df_metrics = pd.DataFrame(metrics)

def plot_confusion_matrix(cm, model_name):
    try:
        df_cm = pd.DataFrame(cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')  # Changed 'd' to '.2f'
        plt.title(f'Confusion Matrix for {model_name}')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error plotting confusion matrix for {model_name}: {e}")

def plot_bar_graph(metric):
    try:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Model', y=metric, data=df_metrics, palette='viridis')
        plt.title(f'{metric} Comparison')
        plt.ylim(0.5, 1.0)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error plotting bar graph for {metric}: {e}")

def plot_histogram(data, title):
    try:
        plt.figure(figsize=(10, 5))
        sns.histplot(data, kde=True, bins=30)
        plt.title(title)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error plotting histogram for {title}: {e}")

st.title('Bankruptcy Prediction Model Comparison')

# Display confusion matrices
for model_name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, model_name)

# Display bar graphs for metrics
metrics_list = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
for metric in metrics_list:
    plot_bar_graph(metric)

# Example data for histograms
hist_data = {
    'Model Accuracy Comparison': [0.85, 0.80, 0.82, 0.88, 0.75],
    'Model Recall Comparison': [0.80, 0.75, 0.78, 0.85, 0.70],
    'Model Precision Comparison': [0.83, 0.78, 0.80, 0.87, 0.72],
    'Model F1 Score Comparison': [0.82, 0.76, 0.79, 0.86, 0.74]
}

# Display histograms
for title, data in hist_data.items():
    plot_histogram(data, title)


# In[ ]:




