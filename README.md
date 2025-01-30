# Intrusion Detection Using Deep Learning (RNN)

## Overview
This project employs machine learning techniques, particularly Recurrent Neural Networks (RNNs), to detect intrusions in a network. The model processes the NSL-KDD dataset, a benchmark dataset for network intrusion detection, to classify network traffic as normal or an attack (DoS, Probe, R2L, U2R).

## Features
- **Dataset Handling**:
  - Uses the NSL-KDD dataset.
  - Performs data preprocessing (cleaning, encoding, and scaling).
  - Supports both binary and multi-class classification tasks.
- **Model Architecture**:
  - Implements RNN using TensorFlow/Keras with LSTM units for temporal pattern recognition.
- **Multi-Model Comparison**:
  - Evaluates various machine learning models, including Decision Trees, Random Forests, MLP, SVM, and Naive Bayes.
- **Visualization**:
  - Visualizes model performance with accuracy plots and confusion matrices.

## Project Structure

Intrusion-Detection-RNN/
├── notebooks/
│   ├── intrusion_detection.ipynb  # Your Jupyter Notebook file
├── data/
│   ├── NSL_KDD_Train.csv         # Training dataset
│   ├── NSL_KDD_Test_21.csv       # Testing dataset
├── README.md                     # Documentation
├── requirements.txt              # List of dependencies


## Installation
1. Clone the repository or download the project files.
2. Ensure you have Python 3.x installed on your system.

## Dependencies

The following libraries are required for this project:
1. pandas: Data manipulation and analysis.
2. numpy: Numerical computing.
3. tensorflow: Building and training the RNN model.
4. scikit-learn: Preprocessing, model evaluation, and alternative machine learning models.
5. matplotlib: Data visualization.
6. seaborn: Enhanced visualization (e.g., bar plots, confusion matrix heatmaps).

## Usage

1. Open the Jupyter Notebook: jupyter notebook notebooks/intrusion_detection.ipynb
2. Run the notebook cells sequentially to:
         -Preprocess the dataset.
         -Train the RNN model.
         -Compare model performance using alternative algorithms.
### Dataset

The project uses the NSL-KDD dataset, stored in the data/ folder. Key preprocessing steps include:

1. One-hot encoding for categorical columns (protocol_type, service, flag).
2. Logarithmic scaling for numerical features like duration, src_bytes, and dst_bytes.
3. Mapping of labels to numerical classes for multi-class classification.

### RNN Architecture

1. Input Layer: Encodes features such as protocol_type, service, and flag.
2. Hidden Layer: 50 LSTM units for sequence learning.
3. Output Layer: 5 units for multi-class classification with softmax activation.

### Model Evaluation

1. Metrics: Accuracy, precision, recall, F1-score, and confusion matrix.
2. Cross-validation: 10-fold cross-validation for RNN evaluation.
3. Comparison: Accuracy of alternative models for specific attack types (e.g., DoS, Probe) is visualized in a bar plot.

### Results

1. Example RNN Test Accuracy: ~85%.
2. Accuracy of other models for specific attack types is visualized in bar plots.


   
