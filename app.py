# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Auto Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .report-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .insight-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    .insight-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
    .insight-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyzer:
    """Main class for data analysis"""
    
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.insights = []
        
    def basic_info(self):
        """Get basic dataset information"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info
    
    def statistics(self):
        """Generate statistics for numeric columns"""
        if self.numeric_cols:
            return self.df[self.numeric_cols].describe().T
        return None
    
    def missing_analysis(self):
        """Analyze missing values"""
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum().values,
            'Missing Percentage': (self.df.isnull().sum() / len(self.df) * 100).values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False)
        return missing_df
    
    def correlation_analysis(self):
        """Calculate correlation matrix"""
        if len(self.numeric_cols) > 1:
            return self.df[self.numeric_cols].corr()
        return None
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        outliers = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
            outliers[col] = {
                'count': outliers_count,
                'percentage': (outliers_count / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        return outliers
    
    def generate_insights(self):
        """Generate automated insights"""
        insights = []
        
        # Dataset size insight
        rows, cols = self.df.shape
        insights.append({
            'type': 'info',
            'message': f"📊 Dataset contains {rows:,} rows and {cols} columns.",
            'category': 'overview'
        })
        
        # Missing values insight
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            high_missing = [col for col in missing_cols if self.df[col].isnull().sum() / len(self.df) > 0.3]
            if high_missing:
                insights.append({
                    'type': 'danger',
                    'message': f"⚠️ Columns with >30% missing values: {', '.join(high_missing)}. Consider dropping or imputing these columns.",
                    'category': 'data_quality'
                })
            else:
                insights.append({
                    'type': 'warning',
                    'message': f"📌 Found missing values in {len(missing_cols)} columns. Consider handling them appropriately.",
                    'category': 'data_quality'
                })
        else:
            insights.append({
                'type': 'good',
                'message': "✅ No missing values detected! Your dataset is complete.",
                'category': 'data_quality'
            })
        
        # Duplicates insight
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            insights.append({
                'type': 'warning',
                'message': f"🔄 Found {duplicates:,} duplicate rows ({duplicates/len(self.df)*100:.1f}% of data). Consider removing duplicates.",
                'category': 'data_quality'
            })
        
        # Numeric columns insights
        if self.numeric_cols:
            insights.append({
                'type': 'info',
                'message': f"📈 Found {len(self.numeric_cols)} numeric columns suitable for statistical analysis.",
                'category': 'features'
            })
            
            # Skewness analysis
            for col in self.numeric_cols[:5]:  # Check first 5 numeric columns
                skewness = self.df[col].skew()
                if abs(skewness) > 1:
                    direction = "right" if skewness > 0 else "left"
                    insights.append({
                        'type': 'warning',
                        'message': f"📊 Column '{col}' is highly skewed ({skewness:.2f}) to the {direction}. Consider transformation for better analysis.",
                        'category': 'statistics'
                    })
        
        # Categorical columns insights
        if self.categorical_cols:
            insights.append({
                'type': 'info',
                'message': f"🏷️ Found {len(self.categorical_cols)} categorical columns for grouping analysis.",
                'category': 'features'
            })
            
            # High cardinality warning
            for col in self.categorical_cols:
                unique_count = self.df[col].nunique()
                if unique_count > 20:
                    insights.append({
                        'type': 'warning',
                        'message': f"⚠️ Column '{col}' has {unique_count} unique values (high cardinality). May need special handling.",
                        'category': 'features'
                    })
                    break
        
        # Correlation insights
        if len(self.numeric_cols) > 1:
            corr_matrix = self.correlation_analysis()
            if corr_matrix is not None:
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
                
                if strong_corrs:
                    for col1, col2, corr in strong_corrs[:3]:
                        direction = "positive" if corr > 0 else "negative"
                        insights.append({
                            'type': 'good' if abs(corr) < 0.9 else 'warning',
                            'message': f"🔗 Strong {direction} correlation ({corr:.2f}) between '{col1}' and '{col2}'.",
                            'category': 'correlations'
                        })
        
        # Outlier insights
        outliers = self.detect_outliers()
        cols_with_outliers = {col: data for col, data in outliers.items() if data['count'] > 0}
        if cols_with_outliers:
            high_outlier_cols = [col for col, data in cols_with_outliers.items() if data['percentage'] > 5]
            if high_outlier_cols:
                insights.append({
                    'type': 'warning',
                    'message': f"📊 Found outliers in {', '.join(high_outlier_cols[:3])} (>5% of data). May impact statistical analysis.",
                    'category': 'statistics'
                })
        
        return insights
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Missing value recommendations
        missing_df = self.missing_analysis()
        if not missing_df.empty:
            high_missing = missing_df[missing_df['Missing Percentage'] > 30]
            if not high_missing.empty:
                recommendations.append({
                    'action': 'Remove high-missing columns',
                    'description': f"Columns {list(high_missing['Column'])} have >30% missing values. Consider dropping them or using advanced imputation.",
                    'priority': 'High'
                })
            else:
                recommendations.append({
                    'action': 'Handle missing values',
                    'description': 'Use mean/median imputation for numeric columns or mode for categorical columns.',
                    'priority': 'Medium'
                })
        
        # Duplicate recommendations
        if self.df.duplicated().sum() > 0:
            recommendations.append({
                'action': 'Remove duplicates',
                'description': f"Remove {self.df.duplicated().sum():,} duplicate rows to avoid bias in analysis.",
                'priority': 'High'
            })
        
        # Skewness recommendations
        for col in self.numeric_cols:
            if abs(self.df[col].skew()) > 1:
                recommendations.append({
                    'action': f'Transform {col}',
                    'description': f'Apply log or Box-Cox transformation to reduce skewness (current skew: {self.df[col].skew():.2f}).',
                    'priority': 'Medium'
                })
                break
        
        # Correlation recommendations
        if len(self.numeric_cols) > 1:
            corr_matrix = self.correlation_analysis()
            if corr_matrix is not None:
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.85:
                            recommendations.append({
                                'action': 'Check multicollinearity',
                                'description': f"'{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}' are highly correlated (r={corr_matrix.iloc[i, j]:.2f}). Consider using one for modeling.",
                                'priority': 'Medium'
                            })
                            break
                    break
        
        # General recommendations
        if len(self.numeric_cols) > 0 and len(self.categorical_cols) > 0:
            recommendations.append({
                'action': 'Explore relationships',
                'description': f'Use groupby analysis to explore how {self.categorical_cols[0]} affects {self.numeric_cols[0]}.',
                'priority': 'Low'
            })
        
        recommendations.append({
            'action': 'Feature engineering',
            'description': 'Create new features from existing ones to improve model performance (e.g., ratios, interactions, date parts).',
            'priority': 'Low'
        })
        
        return recommendations

def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_download_link(df, filename, file_format):
    """Create download link for dataframe"""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}.csv</a>'
        return href
    elif file_format == 'html':
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>
            </div>
            <h2>Data Preview</h2>
            {df.head(10).to_html()}
            <h2>Summary Statistics</h2>
            {df.describe().to_html()}
        </body>
        </html>
        """
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">Download {filename}.html</a>'
        return href

def main():
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=80)
        st.title("Auto Data Analyzer")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📤 Upload Data", "📊 Data Overview", "📈 Visualizations", "💡 Insights & Recommendations", "📄 Report Generator"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This tool automatically analyzes your dataset and provides "
            "comprehensive insights, visualizations, and actionable recommendations."
        )
        
        # Dark/Light mode toggle (bonus feature)
        if st.checkbox("🌙 Dark Mode"):
            st.markdown("""
            <style>
                .stApp { background-color: #1e1e1e; color: #ffffff; }
                .report-card { background-color: #2d2d2d; }
            </style>
            """, unsafe_allow_html=True)
    
    # Main content area
    st.title("📊 Automated Data Analysis & Reporting Tool")
    st.markdown("Upload your dataset and get instant insights powered by AI")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Initialize analyzer
            analyzer = DataAnalyzer(df)
            
            # Cache results
            @st.cache_data
            def get_analysis_results():
                return {
                    'basic_info': analyzer.basic_info(),
                    'statistics': analyzer.statistics(),
                    'missing_analysis': analyzer.missing_analysis(),
                    'correlation': analyzer.correlation_analysis(),
                    'outliers': analyzer.detect_outliers(),
                    'insights': analyzer.generate_insights(),
                    'recommendations': analyzer.generate_recommendations()
                }
            
            results = get_analysis_results()
            
            # Page routing
            if page == "📤 Upload Data":
                st.success("✅ File successfully loaded!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Columns", f"{df.shape[1]:,}")
                
                st.subheader("📋 Dataset Preview")
                st.dataframe(df.head(10))
                
                st.subheader("🔍 Data Types & Memory Usage")
                # FIXED: Proper DataFrame creation with same length arrays
                dtype_data = {
                    'Column': df.dtypes.index.tolist(),
                    'Data Type': df.dtypes.values.tolist(),
                    'Non-Null Count': df.count().values.tolist(),
                    'Null Count': df.isnull().sum().values.tolist(),
                    'Memory (MB)': (df.memory_usage(deep=True) / 1024**2).values.tolist()
                }
                dtype_df = pd.DataFrame(dtype_data)
                st.dataframe(dtype_df)
                
                # Download cleaned dataset (bonus feature)
                st.subheader("💾 Download Cleaned Dataset")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Remove Duplicates"):
                        df_clean = df.drop_duplicates()
                        st.success(f"Removed {df.duplicated().sum()} duplicates. {len(df_clean)} rows remaining.")
                        st.markdown(create_download_link(df_clean, "cleaned_dataset", "csv"), unsafe_allow_html=True)
                with col2:
                    if st.button("Fill Missing Values"):
                        df_clean = df.copy()
                        for col in df_clean.columns:
                            if df_clean[col].dtype in ['float64', 'int64']:
                                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                            else:
                                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)
                        st.success("Missing values filled with median/mode!")
                        st.markdown(create_download_link(df_clean, "filled_dataset", "csv"), unsafe_allow_html=True)
            
            elif page == "📊 Data Overview":
                st.header("Dataset Overview")
                
                # Basic information
                with st.expander("📌 Dataset Information", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", results['basic_info']['shape'][0])
                    with col2:
                        st.metric("Total Columns", results['basic_info']['shape'][1])
                    with col3:
                        st.metric("Missing Cells", sum(results['basic_info']['missing_values'].values()))
                    with col4:
                        st.metric("Duplicate Rows", results['basic_info']['duplicates'])
                
                # Missing values analysis
                st.subheader("🔍 Missing Values Analysis")
                missing_df = results['missing_analysis']
                if not missing_df.empty:
                    fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                                 title="Missing Values by Column",
                                 color='Missing Percentage',
                                 color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("🎉 No missing values found in the dataset!")
                
                # Statistical summary
                st.subheader("📊 Statistical Summary")
                if results['statistics'] is not None:
                    st.dataframe(results['statistics'])
                
                # Outlier detection
                with st.expander("📈 Outlier Analysis"):
                    outliers = results['outliers']
                    outlier_data = []
                    for col, data in outliers.items():
                        if data['count'] > 0:
                            outlier_data.append({
                                'Column': col,
                                'Outliers Count': data['count'],
                                'Percentage': f"{data['percentage']:.1f}%"
                            })
                    if outlier_data:
                        st.dataframe(pd.DataFrame(outlier_data))
                    else:
                        st.info("No significant outliers detected!")
            
            elif page == "📈 Visualizations":
                st.header("Data Visualizations")
                
                # Correlation heatmap
                if results['correlation'] is not None:
                    st.subheader("🔥 Correlation Heatmap")
                    fig = px.imshow(results['correlation'], 
                                   text_auto=True, 
                                   aspect="auto",
                                   color_continuous_scale='RdBu_r',
                                   title="Feature Correlations")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numeric columns for correlation analysis (need at least 2).")
                
                # Distribution plots
                st.subheader("📊 Distribution Analysis")
                col_type = st.radio("Select column type:", ["Numerical", "Categorical"])
                
                if col_type == "Numerical" and analyzer.numeric_cols:
                    selected_col = st.selectbox("Select numerical column:", analyzer.numeric_cols)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}",
                                          marginal="box", nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif col_type == "Categorical" and analyzer.categorical_cols:
                    selected_col = st.selectbox("Select categorical column:", analyzer.categorical_cols)
                    value_counts = df[selected_col].value_counts().head(20)
                    
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f"Top Categories in {selected_col}",
                                labels={'x': selected_col, 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No {col_type.lower()} columns found in the dataset.")
                
                # Pair plot for small datasets
                if len(df) < 1000 and len(analyzer.numeric_cols) <= 5 and len(analyzer.numeric_cols) > 1:
                    st.subheader("🔄 Pair Plot Matrix")
                    st.info("Pair plot for small datasets (limited to 5 numeric columns)")
                    fig = px.scatter_matrix(df[analyzer.numeric_cols[:5]], 
                                           title="Pair Plot Matrix",
                                           dimensions=analyzer.numeric_cols[:5])
                    st.plotly_chart(fig, use_container_width=True)
            
            elif page == "💡 Insights & Recommendations":
                st.header("Automated Insights & Recommendations")
                
                # Display insights
                st.subheader("💭 Key Insights")
                for insight in results['insights']:
                    if insight['type'] == 'good':
                        st.markdown(f'<div class="insight-good">{insight["message"]}</div>', unsafe_allow_html=True)
                    elif insight['type'] == 'warning':
                        st.markdown(f'<div class="insight-warning">{insight["message"]}</div>', unsafe_allow_html=True)
                    elif insight['type'] == 'danger':
                        st.markdown(f'<div class="insight-danger">{insight["message"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info(insight["message"])
                
                # Display recommendations
                st.subheader("🎯 Actionable Recommendations")
                for rec in results['recommendations']:
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if rec['priority'] == 'High':
                                st.error(f"⚠️ {rec['priority']} Priority")
                            elif rec['priority'] == 'Medium':
                                st.warning(f"📌 {rec['priority']} Priority")
                            else:
                                st.info(f"💡 {rec['priority']} Priority")
                        with col2:
                            st.markdown(f"**{rec['action']}**")
                            st.caption(rec['description'])
                        st.markdown("---")
            
            elif page == "📄 Report Generator":
                st.header("Generate Comprehensive Report")
                
                # Summary statistics card
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📊 Executive Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
                with col2:
                    st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Key findings
                st.subheader("🔑 Key Findings")
                for insight in results['insights'][:5]:  # Top 5 insights
                    st.markdown(f"- {insight['message']}")
                
                # Report download options
                st.subheader("📥 Download Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Generate HTML Report"):
                        st.markdown(create_download_link(df, f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "html"), 
                                  unsafe_allow_html=True)
                        st.success("✅ Report generated successfully!")
                
                with col2:
                    if st.button("Export Summary as CSV"):
                        summary_df = pd.DataFrame({
                            'Metric': ['Rows', 'Columns', 'Missing Values', 'Duplicate Rows'],
                            'Value': [df.shape[0], df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()]
                        })
                        st.markdown(create_download_link(summary_df, "summary_stats", "csv"), unsafe_allow_html=True)
                
                # Print option
                st.markdown("---")
                if st.button("🖨️ Print Report"):
                    st.markdown("""
                    <script>
                        window.print();
                    </script>
                    """, unsafe_allow_html=True)
    
    else:
        # Show placeholder when no file uploaded
        st.info("👈 Please upload a CSV or Excel file to get started!")
        
        # Example datasets
        with st.expander("📚 Try with example datasets"):
            st.markdown("""
            - **Customer Sales Dataset**: Customer purchasing behavior
            - **Employee Performance Data**: HR analytics
            - **House Prices Dataset**: Real estate analysis
            - **Medical Patient Data**: Healthcare analytics
            - **Product Sales Data**: E-commerce with anomalies
            
            Upload your own data or download the sample datasets provided above to test the tool!
            """)

if __name__ == "__main__":
    main()
