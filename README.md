## 22035573_Final-Project_Stock-Market
# Stock Market Prediction: A Study on Ensemble Techniques with Machine Learning and Deep Learning Models

The project aims to compare the performance of ensemble models with individual machine learning models. The models that were considered in this project were Random Forest (RF), Support Vector Machine (SVM), and Long Short-Term Memory (LSTM). The other aim of this study is to determine which of the four models performed best in stock market prediction. The three companies that were considered in this project include BAE Systems Plc, Lockheed Martin Corporation, and Northrop Grumman Corporation. The data set used in this project is quite extensive and covers a period of ten years from the years 2014 to 2024.

## Methodology
### 1. Data Loading :
The datasets used in this project includes the historical stock prices of the three foremost defence and aerospace companies namely BAE Systems Plc (Ticker: BA. L), Lockheed Martin Corporation (Ticker: LMT) and Northrop Grumman Corporation (Ticker: NOC).
10 years stock data was retrieved using 'yfinance' library. Inforamtion and summary statistics of stock data were displayed. 
### 2. Data Visualization:
Closing price, volume traded and moving averages of stock data were plotted. Three moving averages were computed for the closing price: 30-day, 60-day and 100-day.
### 3. Data Pre-processing:
Normalized the 'Close' prices using MinMaxScaler in the range of [0, 1]. The datasets were divided into subsets for testing and training wherein 20% of the data was set aside for testing while 80% was used for training.
### 4. Models:
#### Random Forest:
The ‘randomforest_hyperparameter_tuning’ function was developed to perform hyperparameter tuning of a Random Forest Regressor using GridSearchCV. 
##### Hyperparameters tuned
    param_grid = {
        # Number of trees in the forest
        'n_estimators': [50, 100, 150],
        # Maximum depth of each tree
        'max_depth': [None, 10, 20],
        # Minimum samples required to split a node
        'min_samples_split': [2, 5, 10]
    }
#### Support Vector Machine (SVM):
The ‘svm_hyperparameter_tuning’ function was defined with the objective of performing hyperparameter tuning for an SVM Regressor using GridSearchCV.
##### Hyperparameters tuned
    param_grid = {
        # Regularization parameter
        'C': [1, 10, 100],
        # Kernel coefficient
        'gamma': [0.01, 0.1, 1],
        # Epsilon in the epsilon-SVR model
        'epsilon': [0.1, 0.2, 0.5]
    }
#### Long Short-Term Memory (LSTM):
Input sequences (‘look_back’ = 80 days) for LSTM training were developed using a sliding window approach. Used 'TimeseriesGenerator' to create sequences for LSTM model. The architecture comprised an LSTM layer, one dropout layer as well as a dense layer. The LSTM layer was tuned with the activation function ‘tanh’ in order to process sequential da-ta. The drop-out layer also included a dropout of 0.2 to prevent over-fitting. In this case, the ‘Adam’ optimizer was used while Mean Squared Error (MSE) was used as the loss function with the aim of reducing prediction errors.
##### Hyperparameters tuned
    param_grid = {
        # Number of LSTM units to test
        "lstm_units": [32, 64, 128],
        # Batch sizes to test during training
        "batch_size": [32, 64, 128],
        # Learning rates to evaluate
        "learning_rate": [0.001, 0.0005, 0.0001]
    }
#### Stacking Ensemble Model:
Ridge Regressor as a meta-learner combined predictions from individual models i.e, from RF, SVM, and LSTM.


















