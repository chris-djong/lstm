import os 
import numpy as np
import pandas as pd 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


class LSTM():
    ## Init which creates the actual model and loads the latest weights
    def __init__(self):
        # Tweaking parameters
        self.input_window    = 60  # how many input days we need for the prediction
        self.output_window   = 1   # how many output days we would like - only one for now as we calculate whether we increase by 1 or more 
        self.train_perc      = 0.7  # how many percent of data we would like to be train data
        self.val_perc        = 0.2  # percent used for validation after each batch
        self.test_perc       = 1 - self.train_perc - self.validation_perc
        
        # Initialise all the required arrays
        self.is_train = None
        self.x_train, self.y_train = [], []
        self.x_val, self.y_val = [], []
        self.x_test, self.y_test = [], []
        self.x_predict, self.y_predict = [], []

        # Create model
        self.model = Sequential()
        # return sequences has to be set to true because we added multiple layers
        self.model.add(LSTM(units=32, activation='relu', return_sequences=True, input_shape=(self.input_window, 5)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=64, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        print('Successfully created model:')
        print(self.model.summary())

        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        print('Model has been compiled successfully')

    # Function that scale the inputs and outputs and converts the output to a respective label
    # The current labels are given by -1 for a decrease of 3%, 0 for a hold and +1 for an increase of 3% 
    def normalize_sample(sample):
        print('Normalizing data, sample given by', sample, sample.shape)
        if self.is_train:
            if len(sample) != self.input_window + self.output_window:
                print('Dataset size not equal to input + output size')
            # Normalise the dataset
            sample = (sample - sample[0])/(sample[0])

            # Split into x and y component 
            x_sample = sample[:self.input_window]
            y_sample = sample[self.input_window:self.input_window+self.output_window]['Close']

            # And label it according to the increase in percentage
            mask_bull = y_sample > 0.03
            mask_bear = y_sample < -0.03
            mask_all = y_sample == y_sample
            y_sample[mask_all] = 0
            y_sample[mask_bull] = 1
            y_sample[mask_bear] = -1

            return x_sample, y_sample
        else:
            if len(sample) != self.input_window + self.output_window:
                print('Dataset size not equal to input window')
            # Normalise the dataset
            sample = (sample - sample[0])/(sample[0])
            return sample

    # Check whether the data is complete and perform preprocesing steps
    def verify_data(self, data):
        # Verifiy whether we have enough data / if not return an empty array so that we will simply not loop through it
        if ((self.input_window + self.output_window) > data.shape[0]):
            print('Data verification failed due to insufficient rows.')
            return []
        verified_data = data[['Open', 'Low', 'High', 'Close', 'Volume']]
        return verified_data

    ## Functions which initialises the data or just adds it to the current list in case we have some already
    # Data is a dataframe containing the open, low, close, high and volume of the stock for a consecutive amount of dates
    def add_data(self, data, is_train):
        data = self.verify_data(data)

        # Make sure that we stay in the same mode
        self.is_train = is_train if self.is_train is not None
        if self.is_train != is_train: 
            print('Data has been added already in training mode = ', self.is_train)
            print('Please verifiy data integrity..')
        
        # Add the data to the corresponding array 
        if self.train: 
            train_index = int(floor(self.train_perc*data.shape[0]))
            val_index = int(floor((self.train_perc+self.val_perc)*data.shape[0]))
            for i in range(self.input_window, train_index):
                x_sample, y_sample = self.normalize_sample(data[i-self.input_window:i+self.output_window])
                self.x_train.append(x_sample)
                self.y_train.append(y_sample)
            for i in range(train_index, val_index):
                x_sample, y_sample = self.normalize_sample(data[i-self.input_window:i+self.output_window])
                self.x_val.append(x_sample)
                self.y_val.append(y_sample)
            for i in range(val_index, data.shape[0]):
                x_sample, y_sample = self.normalize_set(data[i-self.input_window:i+self.output_window])
                self.x_test.append(x_sample)
                self.y_test.append(y_sample)
        else: # Otherwise we just add the last 60 values of the dataset as we only need one prediction
            x_sample = self.normalize_sample(data[-self.input_window:])
            self.x_predict.append(x_sample)

    # Function which finalises the input of data, it converts the data to numpy arrays and prints how much data has been added in total
    def finalise_input(self):
        self.x_train, self.y_train     = np.array(self.x_train), np.array(self.y_train)
        self.x_test, self.y_test       = np.array(self.x_test), np.array(self.y_test)
        self.x_predict, self.y_predict = np.array(self.x_predict), np.array(self.y_predict)
        if self.train is None:
            print('Please first add some input to the model using self.add_data()')
        elif self.train:
            print('Finalised training inputs')
            print('Training data has been added')
            print('Batch contains %f sets.', self.x_train.shape[0])
        else:
            print('Finalised to be predicted inputs')
            print('Batch contain %f sets to be predicted.', self.x_predict.shape[0])

    def train_model(self):
        # Create callbacks for the training
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        checkpoint = ModelCheckpoint(filepath='checkpoints/', save_weights_only=True, monitor='val_loss', save_best_only=True)
        reduce_lr  = ReduceLROnPlateau(monitor='val_loss', patience=3, min_lr=0.001, factor=0.4, verbose=2)
        csv_logger = CSVLogger("train_results.csv")

        callbacks = [early_stop, checkpoint, reduce_lr, csv_logger]

        # Fit the model using the specified parameters
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), batch_size=32, epochs=10, callbacks=callbacks)   # dont shuffle because in the end the model is also trained on the initial data and then predicted on consecutive data like the test set is created now 
        
        # And finally evaluate the model performance on the test set
        print("Evaluate on test data")
        self.train_results = self.model.evaluate(self.x_test, self.y_test, batch_size=32)
        print("Final train results of test set evaluation given by:", self.train_results) 


        
        
        
