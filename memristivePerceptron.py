import numpy as np
import random
import matplotlib.pyplot as plt
import pylab





"""

This class gives the value of the memristor conductivity change as a function of conductivity, 
which is placed in the fileName file (csv file).

"""





class setReset:

	#INITIALIZATION OF THE OBJECT
	#fileName is the name of the file that includes the plot of deltaG as a function of G

	def __init__(self,fileName):

		#opens and reads the file
		input=open(fileName,'r')
		s=input.readlines()
		self.xDistribution=[]
		self.yDistribution=[]
		for line in s:
			pair=line.split(',')
			self.xDistribution.append(float(pair[0]))
			self.yDistribution.append(float(pair[1]))
		input.close()





	#RETURNS deltaG BY GIVING THE G VALUE
	def deltaG(self,xValue):

		return pylab.interp(xValue,self.xDistribution,self.yDistribution)







"""

This class defines the perceptron, writes and reads the associations. For more information about the 
utilization consult the README file.

"""





class perceptron:

	# INITIALIZATION OF THE OBJECT
	# NA is the number of neurons of population A
	# NB is the number of neurons of population B
	# bias is the "dummy" input, needed to move the threshold

	def __init__(self,NA=2,NB=1,bias=1,setFileName='set.csv',resetFileName='reset.csv'):

		self.A_memristors=.5*np.ones((NA+1, NB)) #memristor matrix, +1 corresponds to the bias
		self.B_memristors=.5*np.ones((NA+1, NB)) #memristor matrix, +1 corresponds to the bias
		self.NA=NA
		self.NB=NB
		self.bias=bias
		self.errors=[] #saves the error (=1) for the output values different from expected in the reading process

		#settings for the set and reset deltaG=f(G), Pt/TiO2-x/Pt (DOI: 10.1109/WISP.2015.7139171)
		self.set=setReset(setFileName) #in the example reset curve for 2.6V
		self.reset=setReset(resetFileName) #in the example reset curve for 2.6V





	# TRAINING PROCESS: MAKES AN ITERATION AND CHANGES THE RESISTANCE OF THE MEMRISTORS IF NEEDED
	# iteration is the combination of input and expected output, e.g. [np.array([1,1]),np.array([1])]

	def train(self,iteration):
	
		iterationCopy=np.copy(iteration)
		#adds the bias to the input
		iterationCopy[0]=np.lib.pad(iterationCopy[0],(0,1),'constant',constant_values=(self.bias)) 
		unit_step = lambda x: 0 if x <= 0 else 1 #threshold for the action potential
		result=[]
		error=np.zeros(self.NB)
		for i in range(self.NB):
			result.append(np.dot(iterationCopy[0],(self.A_memristors.T[i]-self.B_memristors.T[i])))
			error[i]=iterationCopy[1][i]-unit_step(result[-1])
			er=error[i]
			if er<0:
				er=-1.
			elif er>0:
				er=1.
			else:
				er=0.
			for j in range(self.NA+1):
				if abs(self.A_memristors[j][i]-0.9) < abs(self.B_memristors[j][i]-0.9):
					if er < 0.:
						dg = self.set.deltaG(self.B_memristors[j][i])
					else:
						dg = self.reset.deltaG(self.B_memristors[j][i])					
					self.B_memristors[j][i] -= er * dg * iterationCopy[0][j]
				else:
					if er < 0.:
						dg = self.reset.deltaG(self.A_memristors[j][i])
					else:
						dg = self.set.deltaG(self.A_memristors[j][i])	
					self.A_memristors[j][i] += er * dg * iterationCopy[0][j]





	#RETURNS THE OUTPUT GIVING AN INPUT PATTERN ACCORDING TO THE CURRENT STATE OF THE PERCEPTRON
	#pattern is the input, e.g. np.array(1,1)

	def read(self,pattern):
		pattern=np.lib.pad(pattern,(0,1),'constant',constant_values=(self.bias)) #adds the bias to the input
		unit_step = lambda x: 0 if x <= 0 else 1 #threshold for the action potential
		result=[]
		for i in range(self.NB):
			result.append(unit_step(np.dot(pattern,self.A_memristors.T[i]-self.B_memristors.T[i])))

		return np.array(result)





	# TRAINS THE PERCEPTRON ACCORDING TO GIVEN TRAINING DATA
	# data_training is a list with input values and expected output values, e.g. 
	#	[[np.array([0,0]),np.array([0])],[np.array([1,0]),np.array([1])]]
	# numberIterations is the number of times the perceptron iterates a random choice of the training_data

	def trainMany(self,training_data,numberIterations):

		for i in range(numberIterations): 
			for j in range(len(training_data)):
				iteration = training_data[j] #chooses data randomly for the iteration
				self.train(iteration)





	#RETURNS A LIST OF OUTPUTS GIVING A LIST OF INPUT PATTERNS+LIST OF EXPECTED VALUES
	#patterns is a list of input+expected values, e.g. [[np.array([0,0]),np.array([0])],[np.array([1,0]),np.array([0])]]

	def readMany(self,patterns):

		self.errors = []
		absolute = lambda x: abs(x)
		for pattern in patterns:
			self.errors.append(sum(map(absolute,(self.read(pattern[0])-pattern[1])))) #saves the error





# THIS IS THE CODE THAT IS EXECUTED WHEN RUNNING THIS FILE
# it trains the perceptron to predict if a person with a given height and weight is normal or obese
# to change the logic function uncomment the desired function and comment the remaining logic functions.

if __name__ == "__main__":

	#training dataset [height,weight] -> normal or obese
	training_data = [[np.array([1.7,1.0]),np.array([1])],[np.array([1.75,.65]),np.array([0])],
		[np.array([1.8,.93]),np.array([1])],[np.array([1.68,.66]),np.array([0])],
		[np.array([1.82,.76]),np.array([0])],[np.array([1.86,1.11]),np.array([1])],
		[np.array([1.78,.9]),np.array([1])],[np.array([1.9,.8]),np.array([0])],
		[np.array([1.91,.88]),np.array([0])],[np.array([1.81,.94]),np.array([1])],
		[np.array([1.98,.97]),np.array([0])],[np.array([1.84,.98]),np.array([1])],
		[np.array([1.82,.71]),np.array([0])],[np.array([1.69,.85]),np.array([1])],
		[np.array([1.88,.75]),np.array([0])]]

	#test dataset [height,weight] -> normal or obese
	test_data = [[np.array([1.72,.64]),np.array([0])],[np.array([1.76,1.]),np.array([1])],
		[np.array([1.81,.76]),np.array([0])],[np.array([1.67,.91]),np.array([1])],
		[np.array([1.83,.97]),np.array([1])],[np.array([1.84,.78]),np.array([0])],
		[np.array([1.77,.68]),np.array([0])],[np.array([1.92,1.2]),np.array([1])],
		[np.array([1.89,1.4]),np.array([1])],[np.array([1.82,.74]),np.array([0])],
		[np.array([2.,.97]),np.array([0])],[np.array([1.83,.71]),np.array([0])],
		[np.array([1.81,.91]),np.array([1])],[np.array([1.7,.65]),np.array([0])],
		[np.array([1.87,1.01]),np.array([1])]]

	network = perceptron() #if using in the command line please write "network = perceptron.perceptron()"
	network.readMany(test_data)
	beforeTrain = network.errors
	network.trainMany(training_data,1)
	network.readMany(test_data)
	afterTrain = network.errors

	#result shown on the terminal
	print '\nA_memristors'
	print network.A_memristors,'\n'
	print 'A_memristors'
	print network.B_memristors,'\n'
	print 'A_memristors - B_memristors'
	print network.A_memristors - network.B_memristors,'\n'

	#plot
	plt.subplot(121)
	plt.bar(range(len(beforeTrain)),beforeTrain,color='r')
	plt.ylim([-.1,1.1])
	plt.xlabel('test dataset index')
	plt.ylabel('error')
	plt.title('Before training')
	plt.subplot(122)
	plt.bar(range(len(afterTrain)),afterTrain)
	plt.xlabel('test dataset index')
	plt.ylim([-.1,1.1])
	plt.title('After training')
	plt.show()