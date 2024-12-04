# FedML Research

## Overview
This repository focuses on Federated Machine Learning (FedML) and centralized models to protect sensitive data in mental health domains. The goal is to analyze factors impacting mental health outcomes while safeguarding user privacy.

## Table of Contents
1. [Datasets](#datasets)
2. [Research Papers](#research-papers)
3. [Code](#code)
4. [Questionnaire](#questionnaire)
5. [Privacy Considerations](#privacy-considerations)
6. [Results](#results)
7. [Contributors](#contributors)
8. [License](#license)

## Datasets
Details about the datasets used in this project can be found in the `datasets` directory. These datasets are crucial for training and evaluating our models.

## Research Papers
A collection of relevant research papers is available in the `ReferencePapers` directory. These papers provide a theoretical foundation for the techniques and methodologies employed in this project.

## Code
The `Code` directory contains the implementation of both centralized and federated learning models. It includes scripts for data preprocessing, model training, and evaluation.

## Centralized and Federated Learning Models

### Centralized Models
Centralized learning involves aggregating all the data to a central server and training the model on this combined dataset. This approach is straightforward but poses significant privacy risks, especially with sensitive data.

1. **Model Architecture**:
   - **Logistic Regression**: Used for basic binary classification tasks.
   - **Neural Networks**: Utilized for more complex classification tasks, involving dense layers and activation functions like ReLU and softmax.

2. **Training Procedure**:
   - Data is collected from various sources and centralized.
   - Standard preprocessing techniques are applied.
   - The model is trained using traditional gradient descent methods.
   - Performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

### Federated Learning Models
Federated learning (FedML) allows model training across multiple decentralized devices holding local data samples, without exchanging them. This method enhances privacy by keeping raw data on the devices.

1. **Model Architecture**:
   - Similar architectures as centralized models, including logistic regression and neural networks.
   - Models are designed to be lightweight to accommodate varying device capabilities.

2. **Training Procedure**:
   - **Federated Averaging (FedAvg)**: The primary algorithm used. It involves the following steps:
     1. Each client (device) trains a local model on its own data.
     2. Local models' updates are sent to a central server.
     3. The server aggregates these updates to form a global model.
     4. The global model is sent back to the clients, and the process repeats.
   - This iterative process continues until convergence, ensuring that data remains decentralized.

3. **Privacy Mechanisms**:
   - Differential Privacy: Adds noise to the model updates to prevent leakage of individual data points.
   - Secure Aggregation: Ensures that individual updates are not revealed during aggregation.

### Evaluation and Results
- Models are evaluated on metrics such as accuracy, loss, and privacy guarantees.
- Federated models typically show slightly lower performance compared to centralized models due to data heterogeneity but offer significant privacy advantages.

### Implementation Details
- **Libraries and Tools**: TensorFlow Federated, PySyft.
- **Frameworks**: Keras for building neural networks, PyTorch for flexible model experimentation.

## Questionnaire
The project employs a specific questionnaire (e.g., GHQ-20) to gather mental health data. Details about the questionnaire are included in this section.

## Privacy Considerations
To ensure privacy, Federated Machine Learning (FedML) is used. This technique allows model training across multiple devices without centralizing sensitive data.

## Results
The results of our experiments and model evaluations are documented here. This includes performance metrics and visualizations.

## Contributors
- [Anuj Bohra](https://github.com/anujbohra23)
- [Kushal Vadodaria](https://github.com/kushal-vadodaria)

## License
This project is licensed under the MIT License.
