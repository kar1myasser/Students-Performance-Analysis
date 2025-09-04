# Student Performance Analysis

## Project Overview
This project analyzes student performance data from the UCI ML Repository to identify early warning signals for academic underperformance and provide actionable recommendations for universities.

## Dataset
- **Source**: UCI ML Repository - Student Performance Dataset
- **License**: CC BY 4.0
- **Size**: 649 rows, 30+ features
- **Subjects**: Mathematics and Portuguese
- **URL**: https://archive.ics.uci.edu/dataset/320/student+performance

## Project Structure
```
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and processed datasets
├── notebooks/                  # Jupyter notebooks for analysis (4 notebooks)
│   ├── 01_data_preparation.ipynb       # Data cleaning and preprocessing
│   ├── 02_eda_visualization.ipynb      # Exploratory data analysis  
│   ├── 03_unsupervised_learning.ipynb  # K-means clustering analysis
│   └── 04_supervised_learning.ipynb    # Classification model training
├── reports/                    # Technical reports and visualizations
│   ├── data_quality_report.md          # Data quality assessment
│   ├── technical_report.md             # Comprehensive technical report
│   ├── *.pdf                          # PDF versions of reports
│   └── *.png                          # Generated visualizations
├── src/                        # Source code modules
│   ├── data_processing.py              # Data processing utilities
│   ├── model_utils.py                  # Model training utilities
│   ├── visualization.py               # Visualization functions
│   └── __init__.py                     # Package initialization
├── screenshots/                # Project screenshots and images
├── .vscode/                   # VS Code configuration
├── .gitignore                 # Git ignore rules
├── COPILOT_GUIDELINES.md      # Development guidelines
├── download_dataset.py        # Dataset download script
├── environment.yml            # Conda environment configuration
├── project_requirements.md    # Project requirements document
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Open and run notebooks in sequential order (01 through 05)
2. Results and visualizations will be generated within notebooks
3. Export final report and presentation materials

## Key Deliverables
- Data Quality Assessment
- Exploratory Data Analysis with Visualizations
- Student Behavioral Segmentation (K-Means Clustering)
- Academic Risk Prediction Models (Classification)
- Technical Report with Insights and Recommendations
- Presentation Slides


## Authors
Karim Yasser

