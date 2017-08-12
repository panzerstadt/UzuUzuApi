# 2017 August 06, Tang Li Qun
# Project Standy : Uzu Uzu Index modelling
# 4 layer regression multi layer perceptron (MLP) for modelling the uzu uzu index (UUI) used in project Standy
# feel free to play around with it!

# matrix processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy

# neural network pieces
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization

# neural network metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# misc
import os, pickle

# reference
print(__file__)
directory = os.path.dirname(__file__)

# main
def hackathon_preprocess(dataset):
    """
    extremely shitty, only takes one form of input csv
    :param dataset: dirty dataset
    :return: cleaned dataset
    """
    dataset = dataset.ix[:, 2:]  # takes out first column (previous index number) and second column (datse)
    dataset_a = dataset.ix[:, 0]  # +below: takes out atmospheric pressure
    dataset_b = dataset.ix[:, 2:]
    dataset = pd.concat([dataset_a, dataset_b], axis=1)

    dataset.replace(to_replace=['10-', '0+'], value=0, inplace=True)
    dataset.replace(to_replace=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], value=0, inplace=True)
    dataset.replace(to_replace=['Sat', 'Sun'], value=1, inplace=True)
    dataset = dataset.fillna(value=0)

    return dataset

def uzu_uzu_index_predictor(train_dataset=None, new_input=False, load_saved_model=True, save_model=True, plot_results=False):
    """
    defaults to loading pretrained model. if load_saved_model is False, it will retrain on the dataset in the accompanying folder.
    either pretrained or new NN will be used to predict the set of new inputs and return an uzu uzu index

    :param train_dataset: FUTURE - feed dataset from outside
    :param new_input: input list for prediction (refer to uui_calculator for details)
    :param load_saved_model: default = True. if False, will retrain model
    :param save_model: default = True. will only save model if load_saved_model is False and save_model is True
    :param plot_results: use matplotlib to plot the results as a graph (for checking accuracy during training)
    :return: uzu uzu index
    """


    if new_input is not False:
        #print('receiving input: ', new_input)
        col_labels = ['ind',
                      'datetime',
                      'weekday',
                      'atm_pressure',
                      'cloud_cover',
                      'dew_point',
                      'rain_mm',
                      'rh',
                      'snow_cover',
                      'sunlight_hrs',
                      'temp_celsius',
                      'visibility',
                      'weather_rating',
                      'forest area coverage',
                      'LowPressure']
        new_input.insert(0, 0)
        new_input.insert(0, 0)
        new_input.insert(3, 0)
        new_input = pd.DataFrame(data=[new_input], columns=col_labels)
        new_input = hackathon_preprocess(new_input)
        # show processed inputs
        print('\nthis is your input :')
        print(new_input)


    # if True, load pretrained model
    if load_saved_model:
        # load json and create model
        json_file = open(directory + r'/trained_nn/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(directory + r"/trained_nn/model.h5")
        scaler_x = pickle.load(open(directory + r'/trained_nn/model_datascaler.p', 'rb'))
        print("Loaded model from disk")
        model.compile(loss='mse', optimizer='adam')

        # prediction inputs required
        print('prediction inputs required:')
        #temp = '278,09/08/2016 15:00,Tue,998,6,19.6,0,38,,2.47,36.2,20,2,0.087407407,0.900249511,5.810333102'.split(',')
        print("""day of the week = 'ddd'
        cloud cover
        dew point
        rain
        relative humidity
        snow cover
        sunlight hours
        temperature
        weather visibility
        weather rating
        forest area
        low pressure index""")


    # else do the full training
    else:

        dataset = pd.read_csv(directory + r'/ml_dataset/tokyo_dataset_uui2.csv')
        dataset = hackathon_preprocess(dataset)

        row_len, col_len = dataset.shape
        feature_len = col_len - 1
        print('number of data points:', row_len)
        print('number of features considered', feature_len)


        # fix random seed for reproducibility
        seed = 7
        numpy.random.seed(seed)


        # get train and test ratio
        # split dataset into sets
        # might need padding if the length is different
        ratio = 0.8  # train/test ratio
        length = len(dataset.index)
        split_i = ratio * length

        x_train = dataset.ix[:split_i, :-1]
        y_train = dataset.ix[:split_i, -1]

        x_test = dataset.ix[split_i+1:, :-1]
        y_test = dataset.ix[split_i+1:, -1]


        # turn dataframe into numpy arrays
        # required for sklearn stuff
        x_train = x_train.as_matrix()
        y_train = y_train.as_matrix()

        x_test = x_test.as_matrix()
        y_test = y_test.as_matrix()

        # normalize data
        # fit scaler
        scaler_x = MinMaxScaler().fit(x_train)
        scaler_y = MinMaxScaler().fit(y_train)
        x_train = scaler_x.transform(x_train)
        x_test = scaler_x.transform(x_test)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)

        # save scaler for transforming new datasets
        pickle.dump(scaler_x, open(directory + r'/trained_nn/model_datascaler.p', 'wb'))

        # neural networks to use

        def regression_nn():
            # define regression model
            # create model
            model = Sequential()

            # input layer + 20 neuron hidden layer
            model.add(Dense(20, input_dim=feature_len, kernel_initializer='normal'))
            model.add(Activation('relu'))

            # 30 neuron hidden layer
            model.add(Dense(30, kernel_initializer='normal'))
            #model.add(BatchNormalization())
            model.add(Activation('relu'))

            # final nonlinear bit
            model.add(Dense(1, kernel_initializer='normal'))
            model.add(BatchNormalization())
            model.add(Activation('tanh'))
            # compile
            # adam optimizer is super fast compared to sgd
            model.compile(loss='mse', optimizer='adam')
            return model

        def lstm_embedded_nn():
            """
            NOPE.
            :return:
            """

            # initialize
            model = Sequential()

            model.add(Embedding(feature_len, output_dim=64))
            model.add(LSTM(32))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='tanh'))

            model.compile(loss='mse',
                          optimizer='adam')
            return model


        model = regression_nn()
        # regression model params
        model.fit(x_train, y_train, epochs=500, batch_size=24, verbose=1)
        # simple lstm model params
        #model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=1)


        # make prediction
        predictions = model.predict(x_test)

        # score prediction for regression
        # sklearn accuracy_score only for classification
        test_mse = mean_squared_error(y_test, predictions)
        print('test mse : ', test_mse)

        # score prediction for lstm


        result_pred = scaler_y.inverse_transform(predictions)
        result_truth = scaler_y.inverse_transform(y_test)

        output_pred = [x[0] for x in predictions.tolist()]
        output_truth = y_test.tolist()

        avg_accuracy = []
        for i, _ in enumerate(output_pred):
            guess = output_pred[i]
            truth = output_truth[i]
            #print('%.2f' % guess, ' > ', '%.2f' % truth)
            num_distance = abs(guess - truth)
            accuracy = (1 - abs(num_distance - 0)) * 100
            avg_accuracy.append(accuracy)

        print('average accuracy %.2f%%' %(sum(avg_accuracy) / float(len(avg_accuracy))))

        if plot_results:
            # show in a graph how accurate
            plt.plot(result_pred, color="blue")
            plt.plot(result_truth, color="green")
            plt.show()


    # save trained model
    # serialize model to JSON
    if not load_saved_model:
        if save_model:
            model_json = model.to_json()
            with open(directory + "/trained_nn/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(directory + "/trained_nn/model.h5")
            print("Saved model to disk")


    if new_input is not False:
        # make new prediction
        z_test = scaler_x.transform(new_input)
        predictions = model.predict(z_test)
    else:
        predictions = [[0]]


    predictions = [x[0] for x in predictions.tolist()]
    return predictions



#uui = uzu_uzu_index_predictor(dataset, load_saved_model=True)