import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x )

data_set = pd.read_csv("breast-cancer-wisconsin.csv")
data_set = data_set.replace("?", np.NaN)
data_set.drop(['Code_number'],1,inplace = True)
data_set.fillna(data_set.median(),inplace=True)
data_set["Bare_Nuclei"] = data_set["Bare_Nuclei"].astype(int)

def read_and_divide_into_train_and_test(csv_file):
    global data_set

    data_set_copy = data_set.copy()

    training_inputs = pd.DataFrame.sample(data_set_copy,n=(round(80/100 * data_set.shape[0])), frac=None, replace=False, weights=None, random_state=None, axis=0)

    training_labels = training_inputs.pop("Class")

    training_labels = np.array(training_labels.values)

    training_labels = training_labels.reshape(training_labels.shape[0],-1)

    test_inputs = data_set_copy.drop(training_inputs.index)

    test_labels = test_inputs.pop("Class")

    training_inputs = training_inputs[["Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses"]]

    plt.imshow(training_inputs.corr(), cmap='coolwarm', interpolation='none')

    plt.colorbar()

    plt.xticks(range(len(training_inputs.columns)), training_inputs.columns, rotation=90)

    plt.yticks(range(len(training_inputs.columns)), training_inputs.columns)

    plt.gcf().set_size_inches(8,8)

    labels = training_inputs.corr().values
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            plt.text(x,y,"{:.2f}".format(labels[y,x]),ha = 'center', va='center',color = 'white')
    plt.show()

    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    #calculate test_predictions
    #TODO map each prediction into either 0 or 1
    test_outputs = sigmoid(np.dot(test_inputs,weights))
    test_predictions = np.round(test_outputs)
    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    accuracy = tp / data_set.shape[0]
    # accuracy = tp_count / total number of samples
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    #todo plot loss and accuracy change for each iteration
    plt.plot(accuracy_array)
    plt.ylabel("Accuracy")
    plt.show()
    plt.plot(loss_array)
    plt.ylabel("Loss")
    plt.show()
def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []



    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        input = training_inputs
        #calculate outputs
        #calculate loss
        #calculate tuning
        #update weights
        #run_on_test_set
        training_outputs = np.dot(training_inputs,weights)
        training_outputs = sigmoid(training_outputs)
        loss = training_labels - training_outputs
        tunings = loss * sigmoid_derivative(training_outputs)
        weights += np.dot(np.transpose(training_inputs),tunings)
        accuracy_array.append(run_on_test_set(test_inputs,test_labels,weights))
        loss_array.append(np.mean(loss))
        # you are expected to add each accuracy value into accuracy_array
        # you are expected to find mean value of the loss for plotting purposes and add each value into loss_array





    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()