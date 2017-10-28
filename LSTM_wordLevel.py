import sys
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, LSTM
from random import randint
import numpy as np

if "__name__" != "__main__":

    vocabulary = [str(x) for x in xrange(0, 51)]
    vocabulary.append("END")
    print vocabulary

    x = []
    y = []

    for i in xrange(4096):
        x.append([randint(0, 51)])
        y.append([min(x[i][0] + 1, 51)])

    x = np.asarray(x)
    y = np.asarray(y)
    x_train = [[] for i in xrange(4096)]
    y_train = [[] for i in xrange(4096)]

    for i, j in enumerate(x):
        x_train[i] = list(to_categorical(j, num_classes=52))
    y_train = to_categorical(y, num_classes=52)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print x_train.shape, y_train.shape

    x = []
    y = []

    for i in xrange(512):
        x.append([randint(0, 51)])
        y.append([min(x[i][0] + 1, 51)])

    x = np.asarray(x)
    y = np.asarray(y)
    x_test = [[] for i in xrange(512)]
    y_test = [[] for i in xrange(512)]

    for i, j in enumerate(x):
        x_test[i] = list(to_categorical(j, num_classes=52))
    y_test = to_categorical(y, num_classes=52)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = Sequential()
    model.add(LSTM(52, batch_input_shape=(1, 1, 52)))
    model.add(Dense(52, activation='softmax'))

    # opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2, verbose=1, batch_size=1,
              validation_data=(x_test, y_test))

    print "May test model now, enter -1 to exit"
    while(1):
        num = input('Input first number of Sequence: ')
        sequence = []
        if num < 51 and num > -1:
            while(num != 51):
                sequence.append(vocabulary[num])
                x_in = np.array([to_categorical(np.asarray([num]), num_classes=52)])
                y = model.predict(x_in)
                num = np.argmax(y)
            print sequence

        elif num == -1:
            break

        else:
            print "Number must be between 0 and 50, both inclusive"