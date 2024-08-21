# Predictive Analytics for Diabetes Risk Assessment: A Machine Learning Approach with Interpretable Insights

This project aims to predict diabetes risk using machine learning techniques while providing interpretable insights and personalized recommendations.

## Setup

1. Clone the repository

2. Download the dataset:
   - Get the data from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
   - Create a folder named `data` in the project root
   - Place the downloaded CSV files in the `data` folder

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Anthropic API key:
     ```
     ANTHROPIC_API_KEY=your_api_key_here
     ```

5. Set up Weights & Biases:
   - Get your W&B API key
   - Login via terminal:
     ```
     wandb login
     ```

## Project Structure

- `diabetes-prediction.ipynb`: Jupyter notebook containing:
  - Exploratory Data Analysis (EDA)
  - Data processing
  - Hyperparameter tuning
  - Model and feature selection
  - Interpretability analysis
  - Sensitivity analysis

- `app.py`: Streamlit web application for:
  - Predicting diabetes risk using the trained model
  - Providing context-specific explanations
  - Offering lifestyle recommendations based on predictions

## Running the Project

1. Execute the Jupyter notebook:
   ```
   jupyter notebook diabetes-prediction.ipynb
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Additional Information

- The model artifact and scaler generated from the Jupyter notebook is used by the Streamlit app for predictions.
- The app leverages Claude AI model to provide contextual explanations and personalized lifestyle recommendations.