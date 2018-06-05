import pandas as pd
import numpy as np
from sklearn import model_selection
from string import printable
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from pathlib import Path
import json
import os


def print_layers_dims(model):
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)


# GENERAL save model to disk function!
def save_model(model, fileModelJSON, fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)


# GENERAL load model from disk function!
def load_model(fileModelJSON, fileWeights):
    # print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(fileWeights)
    return model


def lstm(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                    dropout=0.2, W_regularizer=W_reg)(main_input)

    # LSTM layer
    lstm = LSTM(lstm_output_size)(emb)
    lstm = Dropout(0.5)(lstm)

    # Output layer (last fully connected layer)
    output = Dense(1, activation='sigmoid', name='output')(lstm)

    # Compile model and define optimizer
    model = Model(input=[main_input], output=[output])
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    df = pd.read_csv('classification_dataset.csv')
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]
    max_len = 75
    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    target = np.array(df.isMalicious)

    X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=0.25,
                                                                                  random_state=33)
    epochs = 3
    batch_size = 32

    model = lstm()
    model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, target_test, verbose=1)

    print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
    print_layers_dims(model)
    model_name = "deeplearning_LSTM"
    save_model(model, model_name + ".json",  model_name + ".h5")
