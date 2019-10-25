import sys
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import utils


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_model(look_back):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    return model


def train_test(market, market_values, epochs, model_weights, show_plot=True):
    """ train model and test
    Args
        market: market name
        market_values: a dict of market values, time->value
        epochs: number of epochs
        model_weights: model_weights file
        show_plot: whether plot results
    Returns
        Training RMSE
        Testing RMSE
    """
    market_values_by_t = sorted(market_values.items(), key=lambda x:x[0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array([v for _, v in market_values_by_t]).reshape(-1, 1))
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    look_back = 3
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    
    # training
    model = create_model(look_back)
    if model_weights is not None:
        model.load_weights(model_weights)
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_x, train_y, epochs=epochs, batch_size=32, verbose=1)
        model.save_weights("lstm_many2one_%s.h5" % market)
    
    # testing
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # inverse minmax
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    # compute RMSE
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:,0]))
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:,0]))
    print('market[%s] train_rmse[%.2f] test_rmse[%.2f] train_mean[%.2f] test_mean[%.2f]' 
          % (market, train_score, test_score, train_y.mean(), test_y.mean()))
    
    if show_plot:
        # plot prediction
        train_predict_plot = np.empty_like(dataset)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
        test_predict_plot = np.empty_like(dataset)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = test_predict
        plt.figure()
        plt.plot(scaler.inverse_transform(dataset), label='Truth')
        plt.plot(train_predict_plot, label='Training predictions')
        plt.plot(test_predict_plot, label='Test predictions')
        plt.ylabel('value', fontsize=20)
        plt.legend(loc=2, prop={'size': 20})
        plt.show()
    return train_score, test_score


if __name__ == '__main__':
    market = sys.argv[1]
    if len(sys.argv) > 2:
        model_weights = sys.argv[2]
    else:
        model_weights = None

    print('Reading data...')
    data = utils.read_json_file('hubinput.json')
    utils.plot_market(data, market)

    market_values = data[market]
    epochs = 50 if len(list(market_values.keys())) < 10000 else 10 # adjust number of epochs by data size
    train_score, test_score = train_test(market, market_values, epochs, model_weights)
    print('training RMSE: %.4f' % train_score)
    print('testing RMSE: %.4f' % test_score)
