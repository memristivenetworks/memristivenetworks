# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""Memristive Willshaw network.

Created for chapter Associative networks and perceptron based on memristors:
fundamentals and algorithmic implementation of the book Handbook of Memristor
Networks, Springer 2018. See Appendix 1 - Python code for memristive-based
Willshaw network.

"""

import numpy as np
import random
import matplotlib.pyplot as plt


class Willshaw:
    """This class defines the Willshaw network, writes and reads the
    associations, and executes an example. For more information about the
    utilization consult the README file.
    """

    def __init__(self, NB=128, NA=128, MB=7, MA=7):
        """INITIALIZATION OF THE OBJECT

        Keyword arguments:
        NB -- number of neurons of population B - input (default 128)
        NA -- number of neurons of population A - output (default 128)
        MB -- number of units of each patten for population B (default 7)
        MA -- number of units of each patten for population A (default 7)
        """

        # matrix NB x NA of memristors
        self.network = np.zeros((NB, NA))
        # lists all associations between B neurons and A neurons
        self.associations = {'B_neurons': [], 'A_neurons': []}
        self.MB = MB
        self.MA = MA
        self.NB = NB
        self.NA = NA

    def write(self, b_neuronList, a_neuronList):
        """ TRAINING PROCESS: WRITES AN ASSOCIATION IN THE NETWORK (MATRIX)

        Keyword arguments:
        b_neuronList -- index list of the neurons of population B taking part
                        in the association
        a_neuronList -- index list of the neurons of population A taking part
                        in the association
        """

        # saves the association in self.associations and self.network if the
        # association is new and the association is MB x MA dimensional
        cond_1 = len(b_neuronList) == self.MB
        cond_2 = len(a_neuronList) == self.MA
        cond_3 = not (sorted(b_neuronList) in self.associations['B_neurons'])
        if cond_1 and cond_2 and cond_3:
            self.associations['B_neurons'].append(sorted(b_neuronList))
            self.associations['A_neurons'].append(sorted(a_neuronList))
            for i in range(self.MB):
                for j in range(self.MA):
                    ind = self.associations['A_neurons'][-1][j]
                    self.network[self.associations['B_neurons'][-1]
                                 [i]][ind] = 1
        else:
            # do nothing if the association is invalid
            print 'invalid association'

    def writeMany(self, numberAssociations):
        """WRITES SEVERAL RANDOM ASSOCIATIONS IN THE MEMRISTIVE MATRIX

        Keyword arguments:
        numberAssociations -- number of random associations to be written

        Return: 2 lists of neurons in the associations
        """
        list_B_neurons = []
        list_A_neurons = []
        i = 0
        while i < numberAssociations:
            b_neuronList = sorted(random.sample(range(self.NB), self.MB))
            a_neuronList = sorted(random.sample(range(self.NA), self.MA))
            # different input patterns required
            if b_neuronList not in list_B_neurons:
                list_B_neurons.append(b_neuronList)
                list_A_neurons.append(a_neuronList)
                i += 1
            else:
                print 'invalid association'
        # uses write method to write in the network
        for i in range(numberAssociations):
            self.write(list_B_neurons[i], list_A_neurons[i])
        return list_B_neurons, list_A_neurons

    def read(self, b_neuronList, threshold):
        """READS THE OUTPUT IN POPULATION B FOR AN INPUT GIVEN FOR POPULATON A

        Keyword arguments:
        b_neuronList -- index list for population B for which an action is
        given threshold -- minimum value for which we have a state 1 in neurons
        of population B

        Return: index list for population A
        """

        b_neuronList = sorted(b_neuronList)
        a_neuronList = []
        for i in range(self.NA):
            sum1 = 0.
            for j in b_neuronList:
                sum1 += self.network[j][i]
                if sum1 >= threshold:
                    a_neuronList.append(i)
        a_neuronList = sorted(a_neuronList)
        return a_neuronList

    def count(self, threshold=None):
        """COUNTS HOW MANY PATTERNS ARE STILL OK

        Keyword arguments:
        threshold -- minimum value for which we have a state 1 in neurons of
                     population A

        Return: the number and average number of associations that are
                retrieved correctly
        """

        sum1 = 0
        errors = []
        if threshold is None:
            threshold = self.MB
        for i in range(len(self.associations['B_neurons'])):
            list_error = sorted(self.read(self.associations['B_neurons'][i],
                                          threshold))
            error = [x for x in list_error if x not in sorted(
                self.associations['A_neurons'][i])]
            if len(error) <= 1:
                sum1 += 1
            else:
                print 'error =', len(error)
            errors += [len(error)]
        return sum1, np.average(errors)


if __name__ == "__main__":
    """THIS IS THE CODE THAT IS EXECUTED WHEN RUNNING THIS FILE
    It creates a willshaw network and stores and retrieves patterns from it.
    It plots the number correctely retrieved patterns and the average number of
    incorrect units in the retrieved patterns, both as a function of number of
    retrieved patterns
    """

    numberSimulations = 1
    PAmax = 250  # Kmax = (NA*NB)/(MA*MB)*ln(2)
    a = []  # list of number of patterns over simulations
    b = []  # list of errors over simulations
    average_capacity = []  # list of capacity over simulations
    for j in range(numberSimulations):
        print '>>>Simulation:', j, '/', numberSimulations
        x = []
        y = []
        capacity = []
        for i in range(PAmax):
            print '#Patterns =', i
            network = Willshaw()  # new network for each i
            beta, alfa = network.writeMany(numberAssociations=i)
            x.append(i)
            out = network.count()
            y.append(out[0])
            capacity.append(out[1])
        x = np.array(x)
        y = np.array(y)
        capacity = np.array(capacity)
        a.append(x)
        b.append(y)
        average_capacity.append(capacity)
    # average  values over numberSimulations
    x = sum(a) / float(numberSimulations)
    y = sum(b) / float(numberSimulations)
    capacity = sum(average_capacity) / float(numberSimulations)
    deviation = map(lambda x: np.std(x), zip(*b))  # deviation area
    # Plots
    plt.figure('correct patterns')
    plt.fill_between(x, y - np.array(deviation), y +
                     np.array(deviation), facecolor='grey')
    plt.plot(x, y)
    plt.xlabel('number of written patterns')
    plt.ylabel('number of correctly retrieved patterns')
    plt.figure('average error')
    plt.bar(x, capacity)
    plt.plot(x, np.ones(PAmax))
    plt.xlabel('number of written patterns')
    plt.ylabel('average number of incorrect units in the retrieved patterns')
    plt.show()
