from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

savedGradient = []
selectedFunction = ''
functions = [
    "Sigmoid",
    "Hyperbloic Tangent",
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

    # To get the number of classes for the output layer
    numberOfClasses = len(dataframe.species.unique())
    neuronsValue.append(numberOfClasses)
    # insert number of features for the input layer
    neuronsValue.insert(0, 5)

    # Combine hidden layers with the output layer and neurons into dictionary
    # hiddenLayers = dict(zip(range(layersValue+1), neuronsValue))
    hiddenLayers = neuronsValue

    # check if bias is selected or not
    if biasCheckBox.get() == 0:
        bias = 0
    else:
        bias = 1

    trainSet, testSet = dataSplitter(dataframe)
    multiLayerPerceptron(epochValue, trainSet, testSet, bias, hiddenLayers, etaValue)


# Split train and test dataframes and shuffle them
def dataSplitter(dataframe):
    dataframe.species.replace({'Adelie': 1, 'Gentoo': 2, 'Chinstrap': 3}, inplace=True)
    # Split each class into separated DF
    class1DataFrame = dataframe.loc[dataframe['species'].isin([1])]
    class2DataFrame = dataframe.loc[dataframe['species'].isin([2])]
    class3DataFrame = dataframe.loc[dataframe['species'].isin([3])]
    # Split DF into train test DF
    class1train, class1test = train_test_split(class1DataFrame, test_size=0.4)
    class2train, class2test = train_test_split(class2DataFrame, test_size=0.4)
    class3train, class3test = train_test_split(class3DataFrame, test_size=0.4)
    # Merge all testSet and trainSet DF together
    testSet = pd.concat([class1test, class2test, class3test])
    trainSet = pd.concat([class1train, class2train, class3train])
    trainSet = shuffle(trainSet)

    trainSet = trainSet.reset_index()
    trainSet = trainSet.drop(columns=['index'])
    testSet = testSet.reset_index()
    testSet = testSet.drop(columns=['index'])

    return trainSet, testSet


def multiLayerPerceptron(epochs, trainSet, testSet, bias, hiddenLayers, etaValue):
    updatedWeightList = list()
    updatedBiasList = list()
    for x in range(epochs):
        for i in range(len(trainSet)):
            features = [[trainSet['bill_length_mm'][i], trainSet['bill_depth_mm'][i], trainSet['flipper_length_mm'][i],
                         trainSet['gender'][i], trainSet['body_mass_g'][i]]]
            actualClass = getActualClass(trainSet['species'][i])

            weightList, biasList, savedNet = forward(features, bias, hiddenLayers, updatedBiasList, updatedWeightList,
                                                     i)
            gradientList = backward(savedNet, weightList, actualClass)
            updatedWeightList, updatedBiasList = updateWeight(gradientList, weightList, hiddenLayers, features,
                                                              savedNet,
                                                              etaValue,
                                                              biasList, bias)

    print('===Train===')
    test(trainSet, updatedWeightList, updatedBiasList, hiddenLayers)
    print('===Test===')
    test(testSet, updatedWeightList, updatedBiasList, hiddenLayers)
    print('\n\n')


def forward(features, bias, hiddenLayers, updatedBiasList, updatedWeightList, layerIndex):
    weightList = list()
    savedNet = list()
    biasList = list()
    # To start with fNet as features at each new training sample
    fNet = features
    for j in range(len(hiddenLayers) - 1):
        # initialize weight matrix and bias matrix for first training sample
        if layerIndex == 0:
            biasMatrix = np.random.rand(1, hiddenLayers[j + 1]) * bias
            weightMatrix = np.random.rand(hiddenLayers[j], hiddenLayers[j + 1])
        # Use updated weight matrix and updated bias matrix from previous train set
        else:
            biasMatrix = updatedBiasList[j]
            weightMatrix = updatedWeightList[j]

        net = np.dot(fNet, weightMatrix) + biasMatrix
        fNet = activationFunction(net)
        savedNet.append(fNet)
        weightList.append(weightMatrix)
        biasList.append(biasMatrix)

    return weightList, biasList, savedNet


def backward(savedNet, savedWeight, actual):
    actual = np.array(actual)
    gradient = np.zeros([])
    reversedSavedNet = savedNet[::-1]
    reversedSavedWeight = savedWeight[::-1]
    for i in range(len(reversedSavedNet)):
        # Output layer equation
        if i == 0:
            gradient = (actual - reversedSavedNet[i]) * derivativeOfActivationFunction(reversedSavedNet[i])
        # hidden layers equation
        else:
            gradient = np.dot(gradient, reversedSavedWeight[i - 1].T,
                              derivativeOfActivationFunction(reversedSavedNet[i]))
        savedGradient.append(gradient)
    return savedGradient


def updateWeight(gradientList, savedWeight, hiddenLayers, feature, savedNet, etaValue, savedBiasMatrix, bias):
    gradientList.reverse()
    updatedWeightList = list()
    updatedBiasList = list()
    feature = np.array([feature])
    for i in range(len(hiddenLayers) - 1):
        newBias = (savedBiasMatrix[i].T + gradientList[i].T * etaValue) * bias
        updatedBiasList.append(newBias.T)

        # Input layer equation
        if i == 0:
            newWeight = savedWeight[i].T + np.dot(gradientList[i].T, feature[i]) * etaValue
        # hidden layers equation
        else:
            newWeight = savedWeight[i].T + np.dot(gradientList[i].T, savedNet[i - 1]) * etaValue

        updatedWeightList.append(newWeight.T)

    return updatedWeightList, updatedBiasList


def derivativeOfActivationFunction(value):
    # Sigmoid
    if selectedFunction == 'Sigmoid':
        return value * (1 - value)
    else:
        return (1 - value) * (1 + value)


def activationFunction(net):
    # Sigmoid
    if selectedFunction == 'Sigmoid':
        return 1 / (1 + np.exp(-net))
    else:
        return (1 - np.exp(-net)) / (1 + np.exp(-net))


def test(dataSet, weightMatrix, biasMatrix, hiddenLayers):
    predictedList = list()
    actualList = list()
    fNet = list()
    for i in range(len(dataSet)):
        features = [[dataSet['bill_length_mm'][i], dataSet['bill_depth_mm'][i],
                     dataSet['flipper_length_mm'][i],
                     dataSet['gender'][i], dataSet['body_mass_g'][i]]]
        features = np.array(features)

        actualClass = getActualClass(dataSet['species'][i])
        actualList.append(np.array(actualClass))

        for j in range(len(hiddenLayers) - 1):
            if j == 0:
                net = np.dot(features, weightMatrix[j]) + biasMatrix[j]
            else:
                net = np.dot(fNet, weightMatrix[j]) + biasMatrix[j]
            fNet = activationFunction(net)

        predictedList.append(fNet)

    predictedList = selectPredictedClass(predictedList)

    confMatrix = np.zeros((3, 3))
    totalTruePos = 0
    # for loop to save the old confusion matrix results
    # with the new confusion matrix from the next actual and predicted class
    for d in range(len(dataSet)):
        result, truePos = confusionMatrix(list(predictedList[d]), list(actualList[d]))
        cm = np.add(result, confMatrix)
        totalTruePos += truePos
    print(confMatrix)
    print('Accuracy:', totalTruePos / len(dataSet) * 100, '%')


def getActualClass(selectedClass):
    if selectedClass == 1:
        actualClass = [1, 0, 0]
    elif selectedClass == 2:
        actualClass = [0, 1, 0]
    else:
        actualClass = [0, 0, 1]

    return actualClass


def selectPredictedClass(predictedList):
    for x in range(len(predictedList)):
        predictedList[x] = predictedList[x].flatten()
        maxNumber = max(predictedList[x])
        for y in range(len(predictedList[x])):
            if maxNumber == predictedList[x][y]:
                predictedList[x][y] = 1
            else:
                predictedList[x][y] = 0
        predictedList[x] = predictedList[x].astype(int)
    return predictedList


# Create confusion Matrix for actual class and predicted class
def confusionMatrix(predictedList, actualList):
    confMatrix = np.zeros((3, 3))
    truePos = 0
    if actualList == predictedList:
        if predictedList == [1, 0, 0]:
            confMatrix[0][0] = 1
        if predictedList == [0, 1, 0]:
            confMatrix[1][1] = 1
        if predictedList == [0, 0, 1]:
            confMatrix[2][2] = 1
        truePos = 1
    if actualList != predictedList:
        if actualList == [1, 0, 0] and predictedList == [0, 1, 0]:
            confMatrix[0][1] = 1
        elif actualList == [1, 0, 0] and predictedList == [0, 0, 1]:
            confMatrix[0][2] = 1
        elif actualList == [0, 1, 0] and predictedList == [1, 0, 0]:
            confMatrix[1][0] = 1
        elif actualList == [0, 1, 0] and predictedList == [0, 0, 1]:
            confMatrix[1][2] = 1
        elif actualList == [0, 0, 1] and predictedList == [1, 0, 0]:
            confMatrix[2][0] = 1
        elif actualList == [0, 0, 1] and predictedList == [0, 1, 0]:
            confMatrix[2][1] = 1
    return confMatrix, truePos


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
    layersText.set('3')
    layersTextField.pack()

    # Add number of neurons
    neuronsText = StringVar()
    neuronsHeader = Label(main_window, text="Add Neurons").pack()
    neuronsTextField = ttk.Entry(main_window, width=20, textvariable=neuronsText)
    neuronsText.set('7 5 3')
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
