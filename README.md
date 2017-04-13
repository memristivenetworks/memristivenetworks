# The memristive Willshaw network and perceptron (chapter X of book "Memristor network")

This directory includes 2 python codes for simulating the functioning of memristive Willshaw networks and memristive perceptrons. These codes are part of the book "**Memristor Networks**, 2nd edition", chapter X. For all the details, please read the chapter. Below it is given a brief descritpion of the codes. The libraries *numpy*, *random* and *matplotlib* must be installed.



## The memristive Willshaw network

In the memristive Willshaw network code, we first define the class with a matrix `network` with `NA` x `NB` dimension. This matrix defines the conductivity state of each memristor. A conductance state of 1 corresponds to the low resistive state, while a conductance of 0 corresponds to the high resistive state. One needs to initiate two lists for the associations between population A and population B. Thus, an association dictionary is created as well with two entries: '`A_neurons`' and '`B_neurons`'. Each of the 2 entries has empty lists when initiated. This class has 4 methods. The first is the `write` method and allows the writing of only one association. The respective inputs are two lists, one with the index of the population A and the other with the index of the population B participating in the association. The second method is the `read` method, that gives the index of the population B associated with the given input of population A. The third method is `writeMany` that allows one to write a specified number of associations randomly. Finally the fourth method is `count`, which gives the total number of retrieved patterns at the end of the writing process. 

In the if `__name__=='__main__':` function several willshaw networks are written with dimension `NA` = `NB` = 128 and with `MA` = `MB` = 7. The number of retrieved patterns as a function of number of written patterns is plotted at the end, and shown below.

<p align="center">
<img src="https://github.com/danieljosesilva/memristiveWillshawNetworkAndPerceptron/blob/master/images/willshawNetwork.png" height="500">
</p>

## The memristive perceptron

To extract the conductance variation values a class called `setReset` was created. The related object loads the data of the conductance variation as a function of previous conductance from the file defined in the main perceptron object (csv file where the first column is the conductance and the second column if the conductance variation). The perceptron object is created by the `perceptron` class. In the initialization function of the class `perceptron` one has to give four inputs: `NA` which is the number of input neurons, `NB` which is the output neurons, and `bias` which is the constant value discussed in the last section. In the `__init__` function these variable are given with a default values, but can be changed if wished when calling the method. The `__init__` function creates a matrix, called `memristors`, that represents all the pairs of memristors mentioned in the last section. It also creates an empty list, called `errors`, that will save the `errors` for each iteration when in the test reading mode (difference of the output and expected value). 

In the `train` method one iteration with input neurons and expected value(s) is performed to update the memristance of each element. The input value iteration is the combination of active input neurons and expected active output neurons, e.g. `[np.array([1,1]),np.array([1])]`. The `read` method gives the active output neurons for a giving active input neurons, without updating the `memristor` matrix. pattern is the input neurons array, e.g. `np.array([1,1])`. The `trainMany` method trains the perceptron according to a list of pairs of expected active output neurons for active input neurons (`training_data`). Finally, the `readMany` method takes as input a list of patterns, with expected values, as test dataset, with a structure identical to the training dataset, and stores the error in the `errors` attribute. 

In the final if `__name__ == "__main__":` condition it is given an example to predict if a person with a given height and weight is normal or obese, with the corresponding `training_data` and `test_data` list. Finally it is plotted the `errors` of each element of the test dataset, before and after the training process. The plot is shown below.

<p align="center">
<img src="https://github.com/danieljosesilva/memristiveWillshawNetworkAndPerceptron/blob/master/images/perceptron.png" height="500">
</p>