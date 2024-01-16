# Forex Price Prediction using Convolutional and LSTM Neural Networks
In this repo you can find seminar, presentation and a demo, showing how one can leverage the use of deep learning method to predict forex prices.


This project utilizes a combination of convolutional and LSTM neural networks to predict forex prices. Specifically, it
focuses on predicting the `EURUSD` currency pair's hourly price.

##### There are many ways to tweak the model to make it more robust, whether if its DBSCAN,DI,SVM to identify outliers and prevent the model from trading while there are sudden changes or by adding more features, setting up threshold for certain actions. Also, you can incorporate attention mechanism like i did in [here](https://github.com/Netanelshoshan/freqAI-LSTM) .


## Features

- Retrieves historical forex data using the MetaTrader5 Python API.
- Utilizes a Conv1D-LSTM architecture for sequence prediction.
- Scales data using MinMaxScaler for better training.
- Plots and visualizes the actual vs. predicted prices.
- Evaluates model performance using metrics like RMSE, MSE, and R2 score.
- Saves the trained model in the ONNX format.

## Requirements
1. Ensure that you have MetaTrader5 installed on your machine and that the necessary configurations for API access are
   set.
2. The script will fetch the required historical data, preprocess it, train the model, evaluate its performance, plot
   results, and save the trained model in ONNX format.

## Model Architecture

- 1D Convolutional layer with 256 filters, kernel size of 2, and ReLU activation.
- MaxPooling layer with pool size 2.
- LSTM layer with 100 units, return sequences set to true.
- Dropout of 0.3 for regularization.
- Another LSTM layer with 100 units.
- Dropout of 0.3 for regularization.
- Dense output layer with sigmoid activation.

## Results

The model was trained for 50 epochs with a batch size of 32. The following results were obtained:
* RMSE         : 0.0009117270571920399
* MSE          : 8.312462268160573e-07
* R2 score     : 0.9667375138591522
