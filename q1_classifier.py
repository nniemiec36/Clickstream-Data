import sys
import argparse
import math
import copy
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score
#Part of code inspired by https://towardsdatascience.com/id3-decision-tree-classifier-from-scratch-in-python-b38ef145fd90
class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    def __init__(self, data='T', children=None):
        self.data = data#attribute value to split on, 'T' for leaf node with value 1 and 'F' for leaf node with value 0
        if children is None:
            children = [-1] * 5 #root node has 5 children nodes
        self.nodes = list(children)#create a list containing nodes representing the children nodes
        
class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""
    def __init__(self, pValueThreshold):
        #may need more values to initialize
        self.pValueThreshold=pValueThreshold
        self.root=None
        self.leavesCount = 0
        self.internalNodesCount = 0

    def fit(self,x,y,node=None,parentNode=None,attributeValue=None):
        #constructs a decision tree using the ID3 algorithm for the training examples x having target values y. 
        #self.pValueThreshold is used for the chi-squared stopping criterion.
        #node represents current node, initial call is from root
        #This function is called recursively

        #Base cases
        if node is None:  # current node is not set
            node = list(x.columns)  # Sets node to be the columns of the training data
        if 0 not in y.unique():  # if all == 1
            currentNode = Node(data = 'T')  # create node(1)
            self.leavesCount += 1  
        elif 1 not in y.unique():  # if all == 0
            currentNode = Node(data = 'F')  # create node(0)
            self.leavesCount += 1  
        elif len(node) == 0:  
            self.leavesCount += 1  
            yCounts = y.value_counts().to_dict() # counts all occurs
            if yCounts.get(1, 0) >= yCounts.get(0, 0):
                currentNode = Node(data = 'T')  
            else:
                currentNode = Node(data = 'F')

        else:
            # Get the attribute with the least entropy
            entropies = {}
            for attr in x[node].columns:
                entropies[attr] = self.calculateEntropy(x[node][attr], y)
            split = min(entropies, key=entropies.get)  # Set attribute to split with min entropy out of all attributes
            node.remove(split)  # remove the attribute from the attributes list

            # calculate the p_value using the chi-squared distribution
            pValue = self.calculateChiSquare(x[split], y)
            

            if pValue < self.pValueThreshold:
                d = split + 1
                currentNode = Node(d)  #create a node with value as best attribute to split on
                self.internalNodesCount += 1 
                for category in range(1, 6):  #for every unique value in the attribute we are splitting
                    x2 = x[x[split] == category]#creates a new training example x2 where the attribute only consists of the current value
                    if x2.empty:  #if there are no rows in this new training example
                        currentNode.nodes[category - 1] = self.createMostCommonYNode(y)#create a node with label = most common value of the target y
                    else:
                        #create a new label dataset which has rows corresponding to the new training example x2
                        y2 = copy.deepcopy(y.loc[x2.index])
                        self.fit(x2, y2, node, parentNode=currentNode, attributeValue=category - 1)
            else:  #the case where the p_value is greater than equal to the significance level
                #create a node with label = most common value of the target y
                currentNode = self.createMostCommonYNode(y)

        if parentNode is None: 
            self.root = currentNode
        else: 
            parentNode.nodes[attributeValue] = currentNode
        return self

    def createMostCommonYNode(self, y):
        self.leavesCount += 1
        yCounts = y.value_counts().to_dict()
        if yCounts.get(1, 0) < yCounts.get(0, 0):
            currentNode = Node(data='F')
        else:
            currentNode = Node(data='T')
        return currentNode

    def calculateEntropy(self,attr_values, target_values):
        entropy = 0
        total = len(attr_values)
        for category in attr_values.unique():  
            cValues = attr_values[attr_values == category] 
            count = float(len(cValues))
            cTargets = target_values.loc[cValues.index]  #get the corresponding target values
            yCounts = cTargets.value_counts().to_dict()  #get counts of corresponding target values
            if len(yCounts) != 2:  #check if the target values contains only one value (either 0 or 1)
                e = 0  
            else:  
                positiveRatio = yCounts[1] / count
                negativeRatio = yCounts[0] / count
                posEnt = positiveRatio * np.log2(positiveRatio)
                negEnt = negativeRatio * np.log2(negativeRatio)
                entropy = -(posEnt + negEnt)
            entropy += (count / total) * entropy
            
        return entropy 

    def calculateChiSquare(self,attribute_vals, targetValues):
        exp = []
        obs = []
        targetValueCounts = targetValues.value_counts().to_dict()
        positiveSet = targetValueCounts.get(1, 0)
        negativeSet = targetValueCounts.get(0, 0)
        total = positiveSet + negativeSet  # Total number of examples
        for category in attribute_vals.unique():  #for each unique value in the attribute
            cValues = attribute_vals[attribute_vals == category]  #get the category values of the current loop
            catCount = float(len(cValues))  #get the count of all the category values
            cTargets = targetValues.loc[cValues.index]  
            yCounts = cTargets.value_counts().to_dict() 
            p, n = yCounts.get(1, 0), yCounts.get(0, 0)  
            
            #calculate the expected and observed number of positives and negatives
            pPrime = positiveSet * catCount / total
            nPrime = negativeSet * catCount / total
            if pPrime != 0:
                exp.append(pPrime)
                obs.append(p)
            if nPrime != 0:
                exp.append(nPrime)
                obs.append(n)
        score, pValue = chisquare(obs, exp)  #calculate the p_value using scipy.stats.chisquare, score not needed for this hw
        return pValue 

    def predict(self, x):
        return pd.Series([self.predictLabel(self.root, i) for i in x[x.columns].values])

    def predictLabel(self, node, row):
        if node.data == 'T':
            return 1
        if node.data == 'F':
            return 0
        index = row[int(node.data) - 1] - 1
        if index > 4:
            index = 4
        return self.predictLabel(node.nodes[index], row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-p', dest='pValueThreshold', action='store', type=float, help='pValueThreshold')
    parser.add_argument('-f1', dest='trainDataset', action='store', type=str, help='trainDataset')
    parser.add_argument('-f2', dest='testDataset', action='store', type=str, help='testDataset')
    parser.add_argument('-o', dest='output_file', action='store', type=str, help='output_file')
    parser.add_argument('-t', dest='decision_tree', action='store', type=str, help='decision_tree')

    args = parser.parse_args()#Parses command arguments to make them easily accessible

    xTrainFile = str(args.trainDataset)
    xTestFile = str(args.testDataset)
    yTrainFile = xTrainFile.replace('feat.csv', 'labs.csv')
    yTestFile = xTestFile.replace('feat.csv', 'labs.csv')

    #Get Dataframes for each file, if we take 0th index we get the series
    xTrain = pd.read_csv(xTrainFile, header=None,sep='\s+')
    xTest = pd.read_csv(xTestFile, header=None,sep='\s+')
    yTrain = pd.read_csv(yTrainFile, header=None,sep='\s+')[0]
    yTest = pd.read_csv(yTestFile, header=None,sep='\s+')[0] 

    model = DecisionTreeClassifier(args.pValueThreshold).fit(xTrain,yTrain)
    with open(args.decision_tree, 'wb') as f:
            pickle.dump(model.root, f)#Sasves tree

    print('Number of internal nodes: ', str(model.internalNodesCount))
    print('Number of leaf nodes: ', str(model.leavesCount))

    yPred = model.predict(xTest)
    yPred.to_csv(args.output_file)   #This writes results to output file

    accuracy = accuracy_score(yTest, yPred)  #calculate accuracy if the target values for test data is given
    print('Model Accuracy: '+ str(accuracy))    
