import pandas as pd
import numpy as np
from sklearn import model_selection
from string import printable
from keras.preprocessing import sequence
from CNNC import CNNC

import warnings
from LSTMC import LSTMC

warnings.filterwarnings("ignore")
_DATA= 'data/'

def run_model(model, url_int_tokens, target, max_len, epochs, batch_size, model_from_file, name):
    X_train, X_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=0.25,
                                                                                  random_state=33)
    print("Running model " + name)
    if model_from_file is None:
        model.train_model(X_train, target_train, epochs=epochs, batch_size=batch_size)
        model.save_model(_DATA + name + ".json", _DATA + name + ".h5")
    else:
        model.load_model(_DATA + name + ".json", _DATA + name + ".h5")
    # loss, accuracy = lstm_model.test_model(X_test, target_test)
    # print("loss " + str(loss))
    # print("accuracy " + str(accuracy))
    model.export_plot()


if __name__ == '__main__':
    df = pd.read_csv('classification_dataset.csv')
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]
    max_len = 75
    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    target = np.array(df.isMalicious)
    epochs = 5
    batch_size = 32

    lstm_model = LSTMC()
    cnn_model = CNNC()
    run_model(cnn_model, url_int_tokens, target, max_len, epochs, batch_size, "lstm_model", "cnn_model")
    run_model(lstm_model, url_int_tokens, target, max_len, epochs, batch_size, "lstm_cnn", "lstm_model")
    # print(lstm_model.predict("https://services.runescape.com-of.top/m=weblogin/a=13/loginform.ws"))
    # print(cnn_model.predict("https://services.runescape.com-of.top/m=weblogin/a=13/loginform.ws"))
    print(lstm_model.predict("https://www.rojadirecta.net/watchnbafinalsstreamfree"))
    print(cnn_model.predict("https://www.rojadirecta.net/watchnbafinalsstreamfree"))




