
import argparse
import math

# Parameters
uniqueWords = set()
actualClass = []
predictedClass = []
uniqueWordsDict = {}
conditionalProbs = {}

# Initialization
def init(trainingFile):
    file = open(trainingFile,'r')
    for line in file:
        newLine = line.split(" ")
        for i in range(2, len(newLine), 2):
            uniqueWords.add(newLine[i])
    conditionalProbs["spam"] = {}
    conditionalProbs["ham"] = {}
    uniqueWordsDict["spam"] = {}
    uniqueWordsDict["ham"] = {}
    for w in uniqueWords:
        uniqueWordsDict["spam"][w] = 0
        uniqueWordsDict["ham"][w] = 0
    file.close()

# Classifier
def classifier(trainingFile):
    totalEMails = 0
    mailTypes = {"spam": 0, "ham": 0}
    file = open(trainingFile,'r')
    for line in file:
        newLine = line.split(" ")
        totalEMails += 1
        mailTypes[newLine[1]] += 1
        for i in range(2, len(newLine), 2):
            uniqueWordsDict[newLine[1]][newLine[i]] += 1
    l = len(uniqueWords)

    # Calculate Conditional Probabilities
    for w, c in uniqueWordsDict["spam"].items():
        conditionalProbs["spam"][w] = float(c + 1) / float(mailTypes["spam"] + l)
    for w, c in uniqueWordsDict["ham"].items():
        conditionalProbs["ham"][w] = float(c + 1) / float(mailTypes["ham"] + l)

# Classify
def classify(testFile, outputFile):
    spamCount = 0
    hamCount = 0
    test = open(testFile,'r')
    output = open(outputFile,'w')

    # Total Counts for Spam and Ham
    for w, count in uniqueWordsDict["spam"].items():
        spamCount += count
    for w, c in uniqueWordsDict["ham"].items():
        hamCount += count
    for line in test:
        hamProb = 0.0
        spamProb = 0.0
        newLine = line.split(" ")
        actualClass.append(newLine[1])
        
        # Using Laplace Smoothing
        for i in range(2, len(newLine), 2):
            spamProb += math.log10(conditionalProbs["spam"][newLine[i]])
            hamProb += math.log10(conditionalProbs["ham"][newLine[i]])
        if spamProb > hamProb:
             predictedClass.append("spam")
             output.write(newLine[0]+ " " + "spam" + "\n")
        else:
            predictedClass.append("ham")
            output.write(newLine[0] + " " + "ham" + "\n")
    test.close()
    output.close()

# Measuring Performance
def performanceMeasurer():
    correctSpam = 0
    incorrectSpam = 0 
    correctHam = 0
    incorrectHam = 0

    for x, y in zip(actualClass, predictedClass):
        if x in "spam" and y in "spam":
            correctSpam += 1
        elif x in "spam" and y in "ham":
            incorrectSpam += 1
        elif x in "ham" and y in "ham":
            correctHam += 1
        else:
            incorrectHam += 1
    
    # Spam and Ham Accuracy
    smoothParam = 15
    p = (float(correctSpam) /float(correctSpam + incorrectSpam)) * 100
    h = (float(correctHam) / float(correctHam + incorrectHam)) * 100
    print("Spam Accuracy: " + str(p))
    print("Ham Accuracy: " + str(h))

if __name__ == "__main__":

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("-f1", required = True, help = "Training File Path")
    argumentParser.add_argument("-f2", required = True, help = "Test File Path")
    argumentParser.add_argument("-o", required = True, help = "Output File Path")

    args = vars(argumentParser.parse_args())
    trainingFile = args['f1']
    testFile = args['f2']
    outputFile = args['o']
    init(trainingFile)
    classifier(trainingFile)
    classify(testFile, outputFile)
    performanceMeasurer()