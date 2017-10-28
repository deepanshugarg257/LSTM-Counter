from __future__ import print_function
import sys
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, LSTM
from random import seed, randint
import numpy as np

def create(x, y, numRange, timesteps):
    # Create Dataset
    string = []
    for i in xrange(2*numRange):
        if i%2:
            string.append(["10"])
        else:
            copy = i/2
            temp = []
            while copy > 0:
                temp.append([str(copy%10)])
                copy /= 10
            string.extend(reversed(temp))

    string = string + [["11"]]*(timesteps-(len(string)%timesteps))
    k = -1
    # for i, j in enumerate(string):
    #     if i%timesteps == 0:
    #         k += 1
    #         x.extend([[]])
    #         y.append([string[min(i+timesteps, len(string)-1)]])
    #     x[k].append(j)

    for i, j in enumerate(string):
        if i >= timesteps:
            k += 1
            x.extend([[]])
            y.append([string[i]])
            x[k].extend(string[i-timesteps:i])

    # for i in x:
    #     print(i)

if "__name__" != "__main__":
    seed()
    vocabulary = {str(x) : x for x in xrange(0, 10)}
    vocabulary[" "] = 10
    vocabulary["E"] = 11
    vocab2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, " ", "E"]
    print(vocabulary)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    numRange = 5000
    numRange = 5000
    timesteps = 10
    epochs = 3

    create(x_train, y_train, numRange, timesteps)
    create(x_test, y_test, numRange, timesteps)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_test = x_test[:1000, :, :]
    y_test = y_test[:1000, :, :]

    rng_state = np.random.get_state()
    np.random.shuffle(x_test)
    np.random.set_state(rng_state)
    np.random.shuffle(y_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    y_train = to_categorical(y_train, num_classes=12)
    y_test = to_categorical(y_test, num_classes=12)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(1, timesteps, 1)))
    model.add(Dense(12, activation='softmax'))

    # opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=1,
              validation_data=(x_test, y_test))

    print("May test model now, enter \"exit\" to exit")
    print("Enter seed sequence of numbers, only %d first characters will be considered" % (timesteps))
    print("--------------------------------------------------------------------")
    while(1):
        string = raw_input('Input Sequence: ')
        sequence = []
        num = 0

        if string == "exit":
            break

        inp = [[vocabulary[i]] for i in string]

        if len(inp) > timesteps:
            inp = inp[:timesteps]

        x_in = np.array([[[11]]*(timesteps - len(inp)) + inp])
        print(x_in)

        while(num != 11):
            y = model.predict(x_in)
            num = np.argmax(y)
            sequence.append(vocab2[num])
            print(vocab2[num], sep="", end="")
            for i in xrange(timesteps-1):
                x_in[0, i, 0] = x_in[0, i+1, 0]
            x_in[0, i+1, 0] = num
        print(sequence)
