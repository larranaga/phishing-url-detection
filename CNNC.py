from string import printable
from keras.models import  Model,  load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers import Input, ELU,  Embedding, \
BatchNormalization, Convolution1D,concatenate
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import CSVLogger
from utils import load_model, save_model
from keras.utils.vis_utils import plot_model


class CNNC:

    def __init__(self, max_len=75, emb_dim=32, max_vocab_len=100, w_reg=regularizers.l2(1e-4)):
        self.max_len = max_len
        self.csv_logger = CSVLogger('CNN_log.csv', append=True, separator=';')
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                        W_regularizer=w_reg)(main_input)
        emb = Dropout(0.25)(emb)

        def sum_1d(X):
            return K.sum(X, axis=1)

        def get_conv_layer(emb, kernel_size=5, filters=256):
            # Conv layer
            conv = Convolution1D(kernel_size=kernel_size, filters=filters, \
                                 border_mode='same')(emb)
            conv = ELU()(conv)

            conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            # conv = BatchNormalization(mode=0)(conv)
            conv = Dropout(0.5)(conv)
            return conv

        # Multiple Conv Layers

        # calling custom conv function from above
        conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
        conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
        conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

        # Fully Connected Layers
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(1024)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization(mode=0)(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        # Output layer (last fully connected layer)
        output = Dense(1, activation='sigmoid', name='output')(hidden2)

        # Compile model and define optimizer
        self.model = Model(input=[main_input], output=[output])
        self.adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def save_model(self, fileModelJSON, fileWeights):
        save_model(self.model, fileModelJSON, fileWeights)

    def load_model(self, fileModelJSON, fileWeights):
        self.model = load_model(fileModelJSON, fileWeights)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, target_train, epochs=5, batch_size=32):
        print("Training CNN model with  " + str(epochs) + " epochs and batches of size " + str(batch_size))
        self.model.fit(x_train, target_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[self.csv_logger])

    def test_model(self, x_test, target_test):
        print("testing CNN model")
        return self.model.evaluate(x_test, target_test, verbose=1)

    def predict(self, x_input):
        url_int_tokens = [[printable.index(x) + 1 for x in x_input if x in printable]]
        X = sequence.pad_sequences(url_int_tokens, maxlen=self.max_len)
        p = self.model.predict(X, batch_size=1)
        return "benign" if p < 0.5 else "malicious"

    def export_plot(self):
        plot_model(self.model, to_file='CNN.png')