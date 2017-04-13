import numpy as np
import random
import matplotlib.pyplot as plt




"""

This class defines the Willshaw network, writes and reads the associations, and executes an example. For more information 
about the utilization consult the README file.

"""





class willshaw:

	# INITIALIZATION OF THE OBJECT
	# NA is the number of neurons of population A
	# NB is the number of neurons of population B
	# MA is the number of units of each patten for population A
	# MB is the number of units of each patten for population B

	def __init__(self, NA=128, NB=128, MA=7, MB=7):

		self.network = np.zeros((NA, NB)) #matrix NA x NB of memristors
		self.associations = {'A_neurons':[], 'B_neurons':[]} #lists all associations between A neurons and B neurons
		self.MA = MA
		self.MB = MB
		self.NA = NA
		self.NB = NB





	# TRAINING PROCESS: WRITES AN ASSOCIATION IN THE NETWORK (MATRIX)
	# a_neuronList is the index list of the neurons of population A taking part in the association
	# b_neuronList is the index list of the neurons of population B taking part in the association

	def write(self, a_neuronList, b_neuronList):

		#saves the association in self.associations and self.network if the association is new and if
		#the association is MA x MB dimensional
		if len(a_neuronList)==self.MA and len(b_neuronList)==self.MB and not (sorted(a_neuronList) in self.associations['A_neurons']):# and not (sorted(b_neuronList) in self.associations['B_neurons']):# and not (sorted(a_neuronList) in self.associations['A_neurons'] and sorted(b_neuronList) in self.associations['B_neurons']):
			self.associations['A_neurons'].append(sorted(a_neuronList))
			self.associations['B_neurons'].append(sorted(b_neuronList))

			for i in range(self.MA):
				for j in range(self.MB):
					self.network[self.associations['A_neurons'][-1][i]][self.associations['B_neurons'][-1][j]] = 1

		#do nothing if the association is invalid
		else:
			print 'association already exists or invalid association'





	# READS THE OUTPUT IN POPULATION B FOR AN INPUT GIVEN FOR POPULATON A
	# a_neuronList is the index list for population A for which an action is given
	# threshold is the minimum value for which we have a state 1 in neurons of population B

	def read(self, a_neuronList,threshold):

		a_neuronList = sorted(a_neuronList)
		b_neuronList = []
		for i in range(self.NB):
			sum=0.
			for j in a_neuronList:
				sum = sum + self.network[j][i]
			if sum+0.5 >= threshold:
				b_neuronList.append(i)

		b_neuronList = sorted(b_neuronList)

		return b_neuronList





	# WRITES SEVERAL RANDOM ASSOCIATIONS IN THE MEMRISTIVE MATRIX
	# numberAssociations is the number of random associations to be written

	def writeMany(self,numberAssociations):

		#generates 2 lists of numberAssociations of associations
		list_A_neurons = []
		list_B_neurons = []
		for i in range(numberAssociations):
			a_neuronList = sorted(random.sample(range(self.NA),self.MA))
			b_neuronList = sorted(random.sample(range(self.NB),self.MB))
			list_A_neurons.append(a_neuronList)
			list_B_neurons.append(b_neuronList)

		#uses write method to write in the network
		for i in range(numberAssociations):
			self.write(a_neuronList=list_A_neurons[i],b_neuronList=list_B_neurons[i])





	# COUNTS HOW MANY PATTERNS ARE CORRECTLY RETRIVED
	# threshold is the minimum value for which we have a state 1 in neurons of population B

	def count(self,threshold):

		sum = 0
		for i in range(len(self.associations['A_neurons'])):
			state = sorted(self.read(a_neuronList=self.associations['A_neurons'][i],threshold=threshold))==sorted(self.associations['B_neurons'][i])
			if state==True:
				sum = sum + 1

		#prints the total written associations and the number of associations that are still ok
		return len(self.associations['A_neurons']), sum 





#THIS IS THE CODE THAT IS EXECUTED WHEN RUNNING THIS FILE
#It plots the number of correctly retrieved patterns as a function of number of written patterns

if __name__ == "__main__":

	numberSimulations = 1#50
	PAmax = 250
	a = []
	b = []
	for j in range(numberSimulations):
		x = []
		y = []
		for i in range(PAmax):
			network = willshaw()
			network.writeMany(numberAssociations=i)
			x.append(i)
			y.append(network.count(7)[1])
		x = np.array(x)
		y = np.array(y)
		a.append(x)
		b.append(y)
	x = sum(a)/float(numberSimulations)
	y = sum(b)/float(numberSimulations)
	deviation = map(lambda x: np.std(x), zip(*b))
	plt.fill_between(x,y-np.array(deviation),y+np.array(deviation),facecolor='grey')
	plt.plot(x,y)
	plt.xlabel('number of written patterns')
	plt.ylabel('number of correctly retrieved patterns')
	print network.network
	plt.show()
#	xlist=np.arange(128)
#	ylist=np.arange(128)
#	X, Y = np.meshgrid(xlist, ylist)
#	Z = network.network[X][Y]
#	plt.colorbar(X,Y,Z)
#	plt.show()