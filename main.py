from tkinter import *
from tkinter import ttk
from tkinter.ttk import Separator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

selectedFunction = ''

functions = [
    "Sigmoid",
    "Hyperbloic Tangent",
]

species = [
    "Adelie",
    "Gentoo",
    "Chinstrap",
]


# Remove null values from gender column and convert it to numerical values
# Normalize all values of all features to range between 0 and 1
def dataNormalization(dataFrame):
    numberOfMales = dataFrame.gender.value_counts().male
    numberOfFemales = dataFrame.gender.value_counts().female
    if numberOfMales > numberOfFemales:
        dataFrame.gender.replace({np.NAN: 'male'}, inplace=True)
    else:
        dataFrame.gender.replace({np.NAN: 'female'}, inplace=True)

    dataFrame.gender.replace({'male': 1, 'female': 0}, inplace=True)

    # Remove species column to apply normalization
    speciesDF = dataFrame[['species']]
    dataFrame = dataFrame.drop(columns=['species'])

    # normalization
    for column in dataFrame.columns:
        dataFrame[column] = dataFrame[column] / dataFrame[column].abs().max()

    # add species column again to dataframe
    frames = [speciesDF, dataFrame]
    dataFrame = pd.concat(frames, axis=1)
    return dataFrame


# get all data from gui
def getDataFromGUI(dataframe):
    selectedFunction = functionValue.get()
    etaValue = float(LearningRateTextField.get())
    epochValue = int(epochTextField.get())
    layersValue = int(layersTextField.get())
    neuronsValue = neuronsTextField.get()
    # Split neurons string in to list of integers
    neuronsValue = neuronsValue.split()
    neuronsValue = [eval(x) for x in neuronsValue]
    # Combine hidden layers and neurons into dictionary
    hiddenLayers = dict(zip(range(layersValue), neuronsValue))
    print('Hidden Layers: ', hiddenLayers)

    if biasCheckBox.get() == 0:
        bias = 0
    else:
        bias = random.random()

    trainSet, testSet = dataSplitter(dataframe)

    forward(trainSet, bias, etaValue, epochValue, hiddenLayers, selectedFunction)
    # test(weightMatrix, testData, feature1, feature2, bias)


# Split train and test dataframes and shuffle them
def dataSplitter(dataframe):
    dataframe.species.replace({'Adelie': 1, 'Gentoo': 2, 'Chinstrap': 3}, inplace=True)
    class1DataFrame = dataframe.loc[dataframe['species'].isin([1])]
    class2DataFrame = dataframe.loc[dataframe['species'].isin([2])]
    class3DataFrame = dataframe.loc[dataframe['species'].isin([3])]
    class1train, class1test = train_test_split(class1DataFrame, test_size=0.4)
    class2train, class2test = train_test_split(class2DataFrame, test_size=0.4)
    class3train, class3test = train_test_split(class3DataFrame, test_size=0.4)
    testSet = pd.concat([class1test, class2test, class3test])
    trainSet = pd.concat([class1train, class2train, class3train])
    testSet = shuffle(testSet)
    trainSet = shuffle(trainSet)

    return trainSet, testSet


def forward(trainSet, bias, etaValue, epochValue, hiddenLayers):
    savedNet = []
    savedWeight = []
    fNet = np.zeros([])
    for x in range(epochValue):

        for i in trainSet.index:

            features = [[trainSet['bill_length_mm'][i], trainSet['bill_depth_mm'][i],
                         trainSet['flipper_length_mm'][i],
                         trainSet['gender'][i], trainSet['body_mass_g'][i]]]
            features = np.array(features)

            actualClass = ''
            if trainSet['species'][i] == 1:
                actualClass = [1, 0, 0]
            elif trainSet['species'][i] == 2:
                actualClass = [0, 1, 0]
            else:
                actualClass = [0, 0, 1]

            for j in hiddenLayers:
                # initialize weight matrix
                if j == 0:
                    weightMatrix = np.random.rand(5, hiddenLayers[j])
                    net = np.dot(features, weightMatrix) + bias
                    print(features.shape, weightMatrix.shape, '=', net.shape)

                else:
                    weightMatrix = np.random.rand(hiddenLayers[j - 1], hiddenLayers[j])
                    net = np.dot(fNet, weightMatrix) + bias
                fNet = activationFunction(net, selectedFunction)
                savedNet.append(fNet)
                savedWeight.append(weightMatrix)
            backward(savedNet, savedWeight, actualClass)


def backward(savedNet, savedWeight, actualClass):
    return 1


def derivativeOfActivationFunction(value, selectedFunction):
    # Sigmoid
    if selectedFunction == 'Sigmoid':
        return value * (1 - value)
    else:
        return (1 - value) * (1 + value)


def activationFunction(net, selectedFunction):
    # Sigmoid
    if selectedFunction == 'Sigmoid':
        return 1 / (1 + np.exp(-net))
    else:
        return (1 - np.exp(-net)) / (1 + np.exp(-net))


if __name__ == '__main__':
    originalDataframe = pd.read_csv(r'penguins.csv')
    originalDataframe = dataNormalization(originalDataframe)

    main_window = Tk()
    main_window.title('Task Three')
    main_window.geometry("512x512")

    # Add number of layers
    layersText = StringVar()
    layersHeader = Label(main_window, text="Number of layer").pack()
    layersTextField = ttk.Entry(main_window, width=20, textvariable=layersText)
    layersText.set('2')
    layersTextField.pack()

    # Add number of neurons
    neuronsText = StringVar()
    neuronsHeader = Label(main_window, text="Add Neurons").pack()
    neuronsTextField = ttk.Entry(main_window, width=20, textvariable=neuronsText)
    neuronsText.set('7 5')
    neuronsTextField.pack()

    # Select Activation function
    functionHeader = Label(main_window, text="Select Activation Function").pack()
    functionValue = StringVar()
    functionValue.set(functions[0])
    feature1DropMenu = OptionMenu(main_window, functionValue, *functions).pack()

    # Add learning rate
    learnText = StringVar()
    LearningRateHeader = Label(main_window, text="Learning Rate").pack()
    LearningRateTextField = ttk.Entry(main_window, width=20, textvariable=learnText)
    learnText.set('0.1')
    LearningRateTextField.pack()

    # Add epoch
    epochText = StringVar()
    epochHeader = Label(main_window, text="Epochs").pack()
    epochTextField = ttk.Entry(main_window, width=20, textvariable=epochText)
    epochText.set('1')
    epochTextField.pack()

    # Select Bias
    biasCheckBox = IntVar()
    checkbox = Checkbutton(main_window, text='Bias', variable=biasCheckBox).pack()

    # Start Classification
    button = Button(main_window, text="Start", command=lambda: getDataFromGUI(originalDataframe)).pack()

    main_window.mainloop()
