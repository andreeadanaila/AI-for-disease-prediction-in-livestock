---

# AI for Disease Prediction in Livestock

This project is a university group assignment developed by four students during the second semester of their second year for the Operating Systems 2 course. It aims to leverage artificial intelligence to predict diseases in livestock based on symptoms.

## Features

* **Symptom-Based Disease Prediction**: Input symptoms to receive probable disease diagnoses.
* **Pre-trained Machine Learning Model**: Utilizes a trained model for accurate predictions.
* **Data Visualization**: Includes tools to visualize data distributions and model performance.
* **Modular Codebase**: Structured for clarity and ease of understanding.

## Project Structure

```
├── dataset/                   # Contains the dataset used for training
├── model/                     # Directory for model-related files
├── visualizations/            # Scripts and outputs for data visualization
├── analysis.ipynb             # Jupyter notebook for exploratory data analysis
├── anthrax.py                 # Script related to anthrax disease prediction
├── disease_classes.json       # JSON file mapping diseases to classes
├── generate_class_mapping.py  # Script to generate class mappings
├── model.py                   # Main model training and evaluation script
├── symptom_vectorizer.pkl     # Pickle file for symptom vectorization
├── trained_model.pkl          # Pickle file of the trained model
├── test_text2.py              # Script for testing the model
├── visualize.py               # Script for generating visualizations
└── README.md                  # Project documentation
```



## Getting Started

### Prerequisites

* Python 3.x
* Required Python libraries listed in `requirements.txt` (not provided in the repository; you may need to create this file based on the imports in the scripts)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/andreeadanaila/AI-for-disease-prediction-in-livestock.git
   cd AI-for-disease-prediction-in-livestock
   ```



2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



### Usage

* **Training the Model**:

```bash
  python model.py
```



* **Testing the Model**:

```bash
  python test_text2.py
```



* **Generating Visualizations**:

```bash
  python visualize.py
```
