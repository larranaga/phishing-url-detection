from keras import regularizers
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Dense, LSTM
from string import printable
from keras.preprocessing import sequence
from utils import save_model, load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger


class LSTMC:

    def __init__(self, max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, w_reg=regularizers.l2(1e-4)):
        super().__init__()
        self.max_len = max_len
        self.csv_logger = CSVLogger('LSTM_log.csv', append=True, separator=';')
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                        dropout=0.2, W_regularizer=w_reg)(main_input)

        # LSTM layer
        lstm = LSTM(lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)

        # Output layer (last fully connected layer)
        output = Dense(1, activation='sigmoid', name='output')(lstm)

        # Compile model and define optimizer
        self.model = Model(input=[main_input], output=[output])
        self.adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def save_model(self, fileModelJSON, fileWeights):
        save_model(self.model, fileModelJSON, fileWeights)

    def load_model(self, fileModelJSON, fileWeights):
        self.model = load_model(fileModelJSON, fileWeights)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, target_train, epochs=3, batch_size=32):
        print("Training LSTM model with  " + str(epochs) + " epochs and batches of size " + str(batch_size))
        self.model.fit(x_train, target_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[self.csv_logger])

    def test_model(self, x_test, target_test):
        print("testing LSTM model")
        return self.model.evaluate(x_test, target_test, verbose=1)

    def predict(self, x_input):
        url_int_tokens = [[printable.index(x) + 1 for x in x_input if x in printable]]
        X = sequence.pad_sequences(url_int_tokens, maxlen=self.max_len)
        p = self.model.predict(X, batch_size=1)
        return "benign" if p < 0.5 else "malicious"

    def export_plot(self):
        plot_model(self.model, to_file='LSTM.png')