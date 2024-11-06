import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef, cohen_kappa_score,
                           balanced_accuracy_score, precision_score, recall_score)
from scipy import stats
import warnings
import os
from sklearn.utils import resample
from jinja2 import Template
import base64
from io import BytesIO
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from difflib import get_close_matches
import chardet
from fuzzywuzzy import fuzz, process  # Add 'process' import
import logging
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

def create_output_directory():
    """Create directory for storing plots if it doesn't exist."""
    output_dir = 'analysis_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_data(file_path):
    """Load data from either Excel or CSV file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.xlsx':
            return pd.ExcelFile(file_path)
        elif file_extension == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

# Load the Excel file
file_path = 'Doctor vs model analysis v3 05.11.2024.xlsx'
data = load_data(file_path)

# Data Import and Exploration
def detect_column_names(actual_columns, required_columns, threshold=80):
    # Create a mapping from required column names to actual column names
    column_mapping = {}
    for req_col in required_columns:
        match = process.extractOne(req_col, actual_columns, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            column_mapping[req_col] = match[0]
        else:
            print(f"Warning: No match found for required column '{req_col}' with sufficient similarity.")
    return column_mapping

def load_and_explore_data(data):
    required_columns = ['actual', 'predicted', 'dp1', 'dp2', 'dp3', 'mp1', 'mp2', 'mp3']  # Define expected columns
    data_dict = {}
    for sheet_name in data.sheet_names:
        df = data.parse(sheet_name)
        print(f"Sheet: {sheet_name}\n")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Data Types:\n", df.dtypes)
        print("Missing Values:\n", df.isnull().sum())

        # Detect and map column names
        column_mapping = detect_column_names(df.columns.tolist(), required_columns)
        df = df.rename(columns=column_mapping)
        data_dict[sheet_name] = df
    return data_dict

data_dict = load_and_explore_data(data)

# Data Cleaning: Missing values imputation (mean or mode based on data type)
def clean_data(data_dict):
    cleaned_data = {}
    for sheet, df in data_dict.items():
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        cleaned_data[sheet] = df
    return cleaned_data

cleaned_data = clean_data(data_dict)

# Descriptive Statistics and Visualization
def descriptive_stats_and_visuals(data_dict):
    output_dir = create_output_directory()
    for sheet, df in data_dict.items():
        print(f"\nDescriptive statistics for {sheet}:")
        print(df.describe())
        
        # Plot histograms and box plots using Plotly
        for col in df.select_dtypes(include=[np.number]).columns:
            # Histogram
            fig_hist = px.histogram(df, x=col, marginal="box", nbins=50,
                                    title=f"{col} Distribution in {sheet}")
            fig_hist.write_html(f"{output_dir}/{sheet}_{col}_histogram.html")
            
            # Box plot
            fig_box = px.box(df, y=col, title=f"{col} Box Plot in {sheet}")
            fig_box.write_html(f"{output_dir}/{sheet}_{col}_boxplot.html")

def comparative_tests(data_dict, alpha=0.05):
    results = []
    sheets = list(data_dict.keys())
    if len(sheets) >= 2:
        df_doctor, df_model = data_dict[sheets[0]], data_dict[sheets[1]]
        for col in df_doctor.select_dtypes(include=[np.number]).columns:
            if col in df_model.columns:
                # Independent t-test and Mann-Whitney U test
                t_stat, p_ttest = ttest_ind(df_doctor[col], df_model[col])
                _, p_mannwhitney = mannwhitneyu(df_doctor[col], df_model[col])

                # Apply Bonferroni correction
                pvals = [p_ttest, p_mannwhitney]
                corrected_pvals = multipletests(pvals, alpha=alpha, method='bonferroni')[1]
                results.append((col, t_stat, corrected_pvals[0], p_mannwhitney, corrected_pvals[1]))
                print(f"{col} | T-test p-value: {corrected_pvals[0]:.4f} | Mann-Whitney p-value: {corrected_pvals[1]:.4f}")
    return results

comparative_test_results = comparative_tests(cleaned_data)

# Correlations and Relationships
def calculate_correlations(data_dict):
    output_dir = create_output_directory()
    sheets = list(data_dict.keys())
    if len(sheets) >= 2:
        df_doctor, df_model = data_dict[sheets[0]], data_dict[sheets[1]]
        
        # Align dataframes by trimming to the shortest length
        min_len = min(len(df_doctor), len(df_model))
        df_doctor = df_doctor.iloc[:min_len]
        df_model = df_model.iloc[:min_len]

        for col in df_doctor.select_dtypes(include=[np.number]).columns:
            if col in df_model.columns:
                correlation = df_doctor[col].corr(df_model[col])
                print(f"Correlation between doctor and model for {col}: {correlation:.4f}")
                
                # Scatter plot with trend line using Plotly
                fig_scatter = px.scatter(x=df_model[col], y=df_doctor[col],
                                         trendline="ols",
                                         labels={'x': 'Model Evaluation', 'y': 'Doctor Evaluation'},
                                         title=f"Scatter Plot with Trend Line for {col} (Doctor vs Model)")
                fig_scatter.write_html(f"{output_dir}/correlation_{col}.html")

calculate_correlations(cleaned_data)

# Regression Analysis
def regression_analysis(data_dict):
    output_dir = create_output_directory()
    sheets = list(data_dict.keys())
    if len(sheets) >= 2:
        df_doctor, df_model = data_dict[sheets[0]], data_dict[sheets[1]]
        
        # Align dataframes to have the same number of rows
        min_len = min(len(df_doctor), len(df_model))
        df_doctor = df_doctor.iloc[:min_len]
        df_model = df_model.iloc[:min_len]

        for col in df_doctor.select_dtypes(include=[np.number]).columns:
            if col in df_model.columns:
                X = df_model[[col]].values.reshape(-1, 1)
                y = df_doctor[col].values
                
                model = LinearRegression()
                model.fit(X, y)
                r_squared = model.score(X, y)
                coef = model.coef_[0]
                intercept = model.intercept_
                print(f"{col} | R-squared: {r_squared:.4f} | Coefficient: {coef:.4f} | Intercept: {intercept:.4f}")
                
                # Regression plot using Plotly
                fig_reg = px.scatter(x=X.flatten(), y=y,
                                     labels={'x': 'Model Evaluation', 'y': 'Doctor Evaluation'},
                                     title=f"Regression Analysis for {col} (Doctor vs Model)")
                fig_reg.add_trace(go.Scatter(x=X.flatten(),
                                             y=model.predict(X),
                                             mode='lines',
                                             name='Regression Line'))
                fig_reg.write_html(f"{output_dir}/regression_{col}.html")

regression_analysis(cleaned_data)

def calculate_accuracy_metrics(data_dict):
    """Calculate various accuracy metrics for doctors and model predictions."""
    sheets = list(data_dict.keys())
    if len(sheets) >= 2:
        df_doctor, df_model = data_dict[sheets[0]], data_dict[sheets[1]]
        
        # Calculate exact match accuracy for each prediction
        doctor_predictions = ['dp1', 'dp2', 'dp3']
        model_predictions = ['mp1', 'mp2', 'mp3']
        
        results = {
            'doctor': {},
            'model': {},
            'comparison': {}
        }
        
        # Calculate accuracy for each prediction
        for dp in doctor_predictions:
            if dp in df_doctor.columns:
                results['doctor'][dp] = df_doctor[dp].value_counts(normalize=True)
        
        for mp in model_predictions:
            if mp in df_model.columns:
                results['model'][mp] = df_model[mp].value_counts(normalize=True)
                
        # Calculate match rates
        for dp, mp in zip(doctor_predictions, model_predictions):
            if dp in df_doctor.columns and mp in df_model.columns:
                exact_match = (df_doctor[dp] == df_model[mp]).mean()
                results['comparison'][f'{dp}_vs_{mp}'] = {
                    'exact_match': exact_match,
                    'similarity': 1 - abs(df_doctor[dp] - df_model[mp]).mean()
                }
        
        return results
    return None

def calculate_medical_metrics(data):
    """Calculate medical statistics with confidence intervals using bootstrapping."""
    if 'actual' in data.columns and 'predicted' in data.columns:
        # Initialize arrays for bootstrap results
        n_iterations = 1000
        sensitivities = np.zeros(n_iterations)
        specificities = np.zeros(n_iterations)
        ppvs = np.zeros(n_iterations)
        npvs = np.zeros(n_iterations)
        aucs = np.zeros(n_iterations)
        
        # Calculate base confusion matrix
        tn, fp, fn, tp = confusion_matrix(data['actual'], data['predicted']).ravel()
        
        # Bootstrap calculations
        for i in range(n_iterations):
            # Generate bootstrap sample
            indices = resample(range(len(data)))
            y_true = data['actual'].iloc[indices]
            y_pred = data['predicted'].iloc[indices]
            
            # Calculate metrics for this iteration
            tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_true, y_pred).ravel()
            sensitivities[i] = tp_i / (tp_i + fn_i)
            specificities[i] = tn_i / (tn_i + fp_i)
            ppvs[i] = tp_i / (tp_i + fp_i)
            npvs[i] = tn_i / (tn_i + fn_i)
            
            # Calculate ROC and AUC for this iteration
            if len(np.unique(y_true)) > 1:  # Check if both classes are present
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                aucs[i] = auc(fpr, tpr)

        # Calculate confidence intervals (95%)
        ci_lower = 2.5
        ci_upper = 97.5
        
        # Main ROC curve calculation
        fpr, tpr, thresholds = roc_curve(data['actual'], data['predicted'])
        roc_auc = auc(fpr, tpr)
        
        return {
            'sensitivity': {
                'value': np.mean(sensitivities),
                'ci': (np.percentile(sensitivities, ci_lower),
                      np.percentile(sensitivities, ci_upper))
            },
            'specificity': {
                'value': np.mean(specificities),
                'ci': (np.percentile(specificities, ci_lower),
                      np.percentile(specificities, ci_upper))
            },
            'ppv': {
                'value': np.mean(ppvs),
                'ci': (np.percentile(ppvs, ci_lower),
                      np.percentile(ppvs, ci_upper))
            },
            'npv': {
                'value': np.mean(npvs),
                'ci': (np.percentile(npvs, ci_lower),
                      np.percentile(npvs, ci_upper))
            },
            'auc': {
                'value': roc_auc,
                'ci': (np.percentile(aucs, ci_lower),
                      np.percentile(aucs, ci_upper))
            },
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'confusion_matrix': {
                'tn': tn, 'fp': fp,
                'fn': fn, 'tp': tp
            }
        }
    return None

def calculate_advanced_metrics(y_true, y_pred):
    """Calculate additional ML evaluation metrics with confidence intervals."""
    n_iterations = 1000
    metrics = {
        'f1': {'values': np.zeros(n_iterations)},
        'mcc': {'values': np.zeros(n_iterations)},
        'kappa': {'values': np.zeros(n_iterations)},
        'balanced_accuracy': {'values': np.zeros(n_iterations)}
    }
    
    # Base calculations
    base_metrics = {
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }
    
    # Bootstrap calculations
    for i in range(n_iterations):
        indices = resample(range(len(y_true)))
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred.iloc[indices]
        
        metrics['f1']['values'][i] = f1_score(y_true_boot, y_pred_boot, average='weighted')
        metrics['mcc']['values'][i] = matthews_corrcoef(y_true_boot, y_pred_boot)
        metrics['kappa']['values'][i] = cohen_kappa_score(y_true_boot, y_pred_boot)
        metrics['balanced_accuracy']['values'][i] = balanced_accuracy_score(y_true_boot, y_pred_boot)
    
    # Calculate confidence intervals
    ci_lower, ci_upper = 2.5, 97.5
    for metric in metrics:
        metrics[metric]['value'] = base_metrics[metric]
        metrics[metric]['ci'] = (
            np.percentile(metrics[metric]['values'], ci_lower),
            np.percentile(metrics[metric]['values'], ci_upper)
        )
    
    return metrics

def enhanced_analysis(data_dict):
    """Perform detailed analysis of predictions with advanced metrics."""
    analysis_results = {}
    
    # Get first sheet's data
    first_sheet_data = data_dict[list(data_dict.keys())[0]]
    
    try:
        # Calculate basic accuracy metrics
        accuracy_metrics = calculate_accuracy_metrics(data_dict)
        if accuracy_metrics:
            analysis_results['accuracy_metrics'] = accuracy_metrics
            
        # Calculate medical metrics if possible
        medical_metrics = None
        if 'actual' in first_sheet_data.columns and 'predicted' in first_sheet_data.columns:
            medical_metrics = calculate_medical_metrics(first_sheet_data)
        else:
            # Try to use alternative column pairs for medical metrics
            possible_pairs = [
                ('dp1', 'mp1'),
                ('doctor_prediction', 'model_prediction'),
                ('doctor_score', 'model_score')
            ]
            
            for doc_col, mod_col in possible_pairs:
                if doc_col in first_sheet_data.columns and mod_col in first_sheet_data.columns:
                    temp_df = pd.DataFrame({
                        'actual': first_sheet_data[doc_col],
                        'predicted': first_sheet_data[mod_col]
                    })
                    medical_metrics = calculate_medical_metrics(temp_df)
                    break
        
        if medical_metrics:
            analysis_results['medical_metrics'] = medical_metrics
            
        # Add statistical tests
        analysis_results['statistical_tests'] = comparative_tests(data_dict)
        
        # Calculate advanced metrics if possible
        if medical_metrics:  # Only if we have medical metrics
            analysis_results['advanced_metrics'] = calculate_advanced_metrics(
                first_sheet_data['actual'] if 'actual' in first_sheet_data.columns else first_sheet_data[doc_col],
                first_sheet_data['predicted'] if 'predicted' in first_sheet_data.columns else first_sheet_data[mod_col]
            )
            
    except Exception as e:
        print(f"Error in enhanced analysis: {str(e)}")
        return None
        
    return analysis_results

# Update generate_visualization_report to use Plotly for ROC curve
def generate_visualization_report(analysis_results):
    """Create enhanced visualizations including ROC curves and confidence intervals."""
    output_dir = create_output_directory()
    
    # Check if analysis_results and medical_metrics exist
    if not analysis_results or 'medical_metrics' not in analysis_results or not analysis_results['medical_metrics']:
        print("Warning: No medical metrics available for visualization")
        return
    
    metrics = analysis_results['medical_metrics']
    
    try:
        # Plot ROC curve with confidence interval using Plotly
        if 'roc_curve' in metrics and all(k in metrics['roc_curve'] for k in ['fpr', 'tpr']):
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=metrics['roc_curve']['fpr'],
                                         y=metrics['roc_curve']['tpr'],
                                         mode='lines',
                                         name='ROC Curve',
                                         line=dict(color='darkorange', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                         mode='lines',
                                         name='Reference Line',
                                         line=dict(color='navy', width=2, dash='dash')))
            fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                                  xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate')
            fig_roc.write_html(f"{output_dir}/roc_curve.html")

        # Performance metrics with confidence intervals
        metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
        if all(metric in metrics for metric in metrics_to_plot):
            plt.figure(figsize=(12, 6))
            values = [metrics[m]['value'] for m in metrics_to_plot]
            ci_lower = [metrics[m]['ci'][0] for m in metrics_to_plot]
            ci_upper = [metrics[m]['ci'][1] for m in metrics_to_plot]
            
            x = np.arange(len(metrics_to_plot))
            plt.bar(x, values, yerr=[np.array(values) - np.array(ci_lower), 
                                    np.array(ci_upper) - np.array(values)],
                    capsize=5)
            plt.xticks(x, metrics_to_plot, rotation=45)
            plt.ylabel('Value')
            plt.title('Performance Metrics with 95% Confidence Intervals')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_metrics.png")
            plt.close()
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        plt.close('all')  # Clean up any open figures

# Modify generate_html_report to embed interactive Plotly plots
def generate_html_report(analysis_results, data_dict):
    """Generate a comprehensive HTML report with all analysis results."""
    
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def fig_to_html_div(fig_html_path):
        try:
            # First try UTF-8
            with open(fig_html_path, 'r', encoding='utf-8') as f:
                fig_html = f.read()
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try with latin-1
                with open(fig_html_path, 'r', encoding='latin-1') as f:
                    fig_html = f.read()
            except UnicodeDecodeError:
                # If both fail, detect encoding
                with open(fig_html_path, 'rb') as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding']
                with open(fig_html_path, 'r', encoding=encoding) as f:
                    fig_html = f.read()
        
        # Extract the div element containing the plot
        start_index = fig_html.find('<div id="')
        end_index = fig_html.rfind('</div>') + 6
        return fig_html[start_index:end_index]

    # Initialize data structures
    plots = {}
    metrics_data = []
    advanced_metrics = []
    accuracy_data = []
    
    try:
        # Process medical metrics
        if analysis_results and 'medical_metrics' in analysis_results:
            metrics = analysis_results['medical_metrics']
            
            # Prepare metrics data
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'value' in metric_data and 'ci' in metric_data:
                    if metric_name not in ['roc_curve', 'confusion_matrix']:
                        metrics_data.append({
                            'name': metric_name.replace('_', ' ').title(),
                            'value': metric_data['value'],
                            'ci': metric_data['ci']
                        })
            
            # Generate ROC Curve plot if available
            if 'roc_curve' in metrics:
                plt.figure(figsize=(10, 8))
                plt.plot(metrics['roc_curve']['fpr'], 
                        metrics['roc_curve']['tpr'], 
                        color='darkorange',
                        label=f"ROC curve (AUC = {metrics['auc']['value']:.2f})")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plots['roc'] = fig_to_base64(plt.gcf())
                plt.close()
        
        # Process accuracy metrics
        if 'accuracy_metrics' in analysis_results:
            acc_metrics = analysis_results['accuracy_metrics']
            if acc_metrics and isinstance(acc_metrics, dict):
                for category, values in acc_metrics.items():
                    if isinstance(values, dict):
                        for metric, value in values.items():
                            accuracy_data.append({
                                'category': category,
                                'metric': metric,
                                'value': value if isinstance(value, (int, float)) else value.mean()
                            })
        
        # Process advanced metrics
        if 'advanced_metrics' in analysis_results:
            adv_metrics = analysis_results['advanced_metrics']
            if adv_metrics and isinstance(adv_metrics, dict):
                for metric_name, metric_data in adv_metrics.items():
                    if isinstance(metric_data, dict) and 'value' in metric_data:
                        advanced_metrics.append({
                            'name': metric_name.replace('_', ' ').title(),
                            'value': metric_data['value'],
                            'ci': metric_data.get('ci', (None, None))
                        })
    
    except Exception as e:
        print(f"Error preparing report data: {str(e)}")
        plt.close('all')  # Clean up any open figures
    
    # Prepare template data
    template_data = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics_data': metrics_data,
        'advanced_metrics': advanced_metrics,
        'accuracy_data': accuracy_data,
        'plots': plots,
        'has_data': bool(metrics_data or advanced_metrics or accuracy_data)
    }
    
    # Update template_data with Plotly plots
    plots = {}
    output_dir = create_output_directory()

    # Collect Plotly plot divs
    plots_files = {
        'roc': f"{output_dir}/roc_curve.html",
    }
    for col in data_dict[list(data_dict.keys())[0]].select_dtypes(include=[np.number]).columns:
        plots_files[f"hist_{col}"] = f"{output_dir}/{list(data_dict.keys())[0]}_{col}_histogram.html"
        plots_files[f"box_{col}"] = f"{output_dir}/{list(data_dict.keys())[0]}_{col}_boxplot.html"
        plots_files[f"scatter_{col}"] = f"{output_dir}/correlation_{col}.html"
        plots_files[f"regression_{col}"] = f"{output_dir}/regression_{col}.html"

    for plot_name, plot_file in plots_files.items():
        if os.path.exists(plot_file):
            plots[plot_name] = fig_to_html_div(plot_file)

    template_data['plots'] = plots
    template_data['has_data'] = bool(metrics_data or advanced_metrics or accuracy_data)

    # Enhanced HTML template with embedded interactive plots
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Report - {{ date }}</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px;
                line-height: 1.6;
                color: #333;
            }
            .section {
                margin: 20px 0;
                padding: 20px;
                background: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .plot-container {
                width: 100%;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f5f5f5;
            }
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                padding: 20px 0;
            }
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>Medical Analysis Report</h1>
        <p>Generated on: {{ date }}</p>

        <!-- Metrics Section -->
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>95% CI</th>
                </tr>
                {% for metric in metrics_data %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ "%.3f"|format(metric.value) }}</td>
                    <td>({{ "%.3f"|format(metric.ci[0]) }}, {{ "%.3f"|format(metric.ci[1]) }})</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <!-- Advanced Metrics Section -->
        {% if advanced_metrics %}
        <div class="section">
            <h2>Advanced Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>95% CI</th>
                </tr>
                {% for metric in advanced_metrics %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ "%.3f"|format(metric.value) }}</td>
                    <td>{% if metric.ci[0] %}({{ "%.3f"|format(metric.ci[0]) }}, {{ "%.3f"|format(metric.ci[1]) }}){% endif %}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <!-- Plots Grid -->
        <div class="section">
            <h2>Visualizations</h2>
            <div class="grid-container">
                {% if plots.roc %}
                <div class="plot-container">
                    <h3>ROC Curve Analysis</h3>
                    {{ plots.roc | safe }}
                </div>
                {% endif %}

                {% for col in plots %}
                    {% if "hist_" in col %}
                    <div class="plot-container">
                        <h3>{{ col.replace('hist_', '').replace('_', ' ').title() }} Distribution</h3>
                        {{ plots[col] | safe }}
                    </div>
                    {% endif %}
                    {% if "box_" in col %}
                    <div class="plot-container">
                        <h3>{{ col.replace('box_', '').replace('_', ' ').title() }} Box Plot</h3>
                        {{ plots[col] | safe }}
                    </div>
                    {% endif %}
                    {% if "scatter_" in col %}
                    <div class="plot-container">
                        <h3>{{ col.replace('scatter_', '').replace('_', ' ').title() }} Correlation</h3>
                        {{ plots[col] | safe }}
                    </div>
                    {% endif %}
                    {% if "regression_" in col %}
                    <div class="plot-container">
                        <h3>{{ col.replace('regression_', '').replace('_', ' ').title() }} Regression</h3>
                        {{ plots[col] | safe }}
                    </div>
                    {% endif %}
                {% endfor %}
            </div>
        </div>

        <!-- Accuracy Metrics Section -->
        {% if accuracy_data %}
        <div class="section">
            <h2>Accuracy Metrics</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for item in accuracy_data %}
                <tr>
                    <td>{{ item.category }}</td>
                    <td>{{ item.metric }}</td>
                    <td>{{ "%.3f"|format(item.value) if item.value is number else item.value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </body>
    </html>
    """

    # Render template
    template = Template(template_string)
    html_report = template.render(**template_data)
    
    # Save report
    report_path = 'analysis_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    return report_path

def export_results(analysis_results, output_path):
    """Export enhanced analysis results to Excel."""
    # Convert nested dictionaries to flat format for DataFrame
    flat_results = {}
    
    if analysis_results and 'medical_metrics' in analysis_results and analysis_results['medical_metrics']:
        metrics = analysis_results['medical_metrics']
        for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'auc']:
            if metric in metrics and metrics[metric] is not None:
                flat_results[f"{metric}_value"] = metrics[metric]['value']
                flat_results[f"{metric}_ci_lower"] = metrics[metric]['ci'][0]
                flat_results[f"{metric}_ci_upper"] = metrics[metric]['ci'][1]
    
    if flat_results:  # Only create and save DataFrame if we have results
        results_df = pd.DataFrame([flat_results])
        results_df.to_excel(output_path, sheet_name='Analysis Results', index=False)
        print(f"Results exported to: {output_path}")
    else:
        print("No results to export - skipping Excel file creation")

# Modified main execution flow
if __name__ == "__main__":
    file_path = 'Doctor vs model analysis v3 05.11.2024.xlsx'
    data = load_data(file_path)
    
    if data is not None:
        data_dict = load_and_explore_data(data)
        cleaned_data = clean_data(data_dict)
        
        # Perform enhanced analysis
        analysis_results = enhanced_analysis(cleaned_data)
        
        if analysis_results:  # Only proceed if we have analysis results
            # Generate visualizations
            generate_visualization_report(analysis_results)
            
            # Generate HTML report
            report_path = generate_html_report(analysis_results, cleaned_data)
            print(f"\nReport generated: {report_path}")
            
            # Export results
            export_results(analysis_results, 'analysis_results.xlsx')
            
            # Print summary
            print("\nAnalysis Summary:")
            print("=================")
            if 'accuracy_metrics' in analysis_results:
                print("\nAccuracy Metrics:")
                print(analysis_results['accuracy_metrics'])
            if 'medical_metrics' in analysis_results:
                print("\nMedical Statistics:")
                print(analysis_results['medical_metrics'])
        else:
            print("No analysis results were generated")
            
        # Original visualization and analysis functions
        descriptive_stats_and_visuals(cleaned_data)
        calculate_correlations(cleaned_data)
        regression_analysis(cleaned_data)
    else:
        print("Failed to load data file")

# Interpretation of Results
print("\nInterpretation:")
print("The results provide insights into the alignment between doctor evaluations and model predictions for various metrics.")
print("Significant differences indicated by statistical tests, correlations, and regression analyses are noted to understand where model predictions deviate from human evaluation.")
print("Detailed summary and visual interpretation are provided to highlight trends and possible improvements in model evaluation alignment.")

class AnalysisStatus(Enum):
    COMPLETE = "complete"
    MISSING = "missing"
    PARTIAL = "partial"
    ERROR = "error"

@dataclass
class AnalysisValidationError:
    error_code: str
    message: str
    field: str

class AnalysisResultHandler:
    def __init__(self):
        self.validation_errors: List[AnalysisValidationError] = []
    
    def validate_analysis_results(self, analysis_results: Optional[Dict[str, Any]]) -> AnalysisStatus:
        if analysis_results is None:
            self.validation_errors.append(
                AnalysisValidationError("NULL_RESULTS", "Analysis results are completely missing", "analysis_results")
            )
            return AnalysisStatus.MISSING
            
        required_fields = {
            'medical_metrics': {'type': dict, 'required_subfields': ['blood_pressure', 'heart_rate']},
            'diagnostic_data': {'type': dict, 'required_subfields': ['diagnosis', 'confidence']},
            'timestamp': {'type': str, 'required_subfields': None}
        }
        
        missing_fields = []
        incomplete_fields = []
        
        for field, requirements in required_fields.items():
            if field not in analysis_results:
                missing_fields.append(field)
                continue
                
            if not isinstance(analysis_results[field], requirements['type']):
                self.validation_errors.append(
                    AnalysisValidationError(
                        "INVALID_TYPE",
                        f"Field {field} has incorrect type. Expected {requirements['type'].__name__}",
                        field
                    )
                )
                continue
                
            if requirements['required_subfields']:
                missing_subfields = [
                    subfield for subfield in requirements['required_subfields']
                    if subfield not in analysis_results[field]
                ]
                if missing_subfields:
                    incomplete_fields.append(f"{field}: missing {', '.join(missing_subfields)}")
        
        if missing_fields:
            self.validation_errors.append(
                AnalysisValidationError(
                    "MISSING_FIELDS",
                    f"Required fields missing: {', '.join(missing_fields)}",
                    "analysis_results"
                )
            )
            return AnalysisStatus.MISSING
            
        if incomplete_fields:
            self.validation_errors.append(
                AnalysisValidationError(
                    "INCOMPLETE_FIELDS",
                    f"Incomplete fields: {'; '.join(incomplete_fields)}",
                    "analysis_results"
                )
            )
            return AnalysisStatus.PARTIAL
            
        return AnalysisStatus.COMPLETE
    
    def process_analysis_results(self, analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        status = self.validate_analysis_results(analysis_results)
        
        if status == AnalysisStatus.MISSING:
            raise ValueError("Cannot process missing analysis results")
            
        if status == AnalysisStatus.ERROR:
            raise ValueError(f"Analysis results validation failed: {self.validation_errors}")
            
        processed_results = {
            'status': status.value,
            'validation_errors': [err.__dict__ for err in self.validation_errors],
            'data': {}
        }
        
        try:
            if status == AnalysisStatus.COMPLETE:
                processed_results['data'] = {
                    'medical_metrics': self._process_medical_metrics(analysis_results['medical_metrics']),
                    'diagnostic_data': self._process_diagnostic_data(analysis_results['diagnostic_data']),
                    'timestamp': analysis_results['timestamp']
                }
            elif status == AnalysisStatus.PARTIAL:
                # Process only available fields
                for field in ['medical_metrics', 'diagnostic_data']:
                    if field in analysis_results:
                        processed_results['data'][field] = getattr(
                            self,
                            f'_process_{field}'
                        )(analysis_results[field])
                
        except Exception as e:
            self.validation_errors.append(
                AnalysisValidationError(
                    "PROCESSING_ERROR",
                    f"Error processing results: {str(e)}",
                    "analysis_results"
                )
            )
            processed_results['status'] = AnalysisStatus.ERROR.value
            
        return processed_results
