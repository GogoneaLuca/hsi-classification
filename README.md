# Hyperspectral Image (HSI) Classification

This repository contains the code for a machine learning classification pipeline designed to process and classify complex Hyperspectral Imaging (HSI) data. It was developed as part of a Kaggle data science competition.

##  Project Overview
Hyperspectral imaging collects and processes information from across the electromagnetic spectrum. The goal of this project is to accurately classify data points based on hundreds of spectral bands. Due to the high dimensionality of the data, the challenge requires robust feature engineering and advanced modeling techniques to prevent overfitting.

##  Tech Stack & Methods
* **Language:** Python
* **Libraries:** PyTorch, LightGBM, Scikit-Learn, Pandas, NumPy, SciPy
* **Data Processing:** * Applied **Savitzky-Golay filters** (`scipy.signal`) to smooth the spectral data and reduce signal noise.
  * Extracted statistical features and reduced dimensionality.
* **Modeling Strategy:**
  * Implemented a **Hybrid Ensemble** approach combining Deep Learning and Gradient Boosting.
  * Developed a **1D-ResNet (PyTorch)** model to extract deep features from the raw spectral signals.
  * Trained a **LightGBM** classifier for the structured tabular data.
  * Configured **Stratified K-Fold Cross-Validation** to ensure the models generalize well, blending their predictions for the final output.

##  How to Use
1. Clone this repository.
2. Ensure you have the required dependencies installed (`pip install torch lightgbm scikit-learn pandas numpy scipy`).
3. Place your training and testing CSV files in a local `data/` folder.
4. Run the Jupyter Notebook cells sequentially.

##  Results
The pipeline successfully optimized the multi-class log-loss metric, achieving a highly competitive score on the Kaggle private leaderboard.
