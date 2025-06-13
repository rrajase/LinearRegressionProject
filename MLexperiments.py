import sqlite3
import csv
import math
import numpy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import linear_model

# Functions to characterize the most common words in a description
# Scrabble Score should capture how "unique" a word is in English
def letterScore(let):
    """letterScore returns the corresponding scrabble point of let
       Arguments: let is a single character
       Return value: the point value of let
    """
    if let in 'aeilnorstu':
        return 1
    elif let in 'dg':
        return 2
    elif let in 'bcmp':
        return 3
    elif let in 'fhvwy':
        return 4
    elif let in 'k':
        return 5
    elif let in 'jx':
        return 8
    elif let in 'qz':
        return 10
    else:
        return 0

def scrabbleScore(S):
    """scrabbleScore returns the number of  scrabble points for a string
       Arguments: S is a string of any length
       Return value: the point value of S
    """
    if len(S) == 0:
        return 0
    else:
        return letterScore(S[0]) + scrabbleScore(S[1:])

# Get the data in a SQL Table

con = sqlite3.connect("imdb.db")
cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS imdb_movies")
cur.execute("CREATE TABLE imdb_movies(id, title, release_date, rating, genre, description, status, language, budget, revenue, country)")

with open('imdb_movies.CSV', newline='', errors='ignore') as csvfile:
  data_reader = csv.reader(csvfile)
  rows = [movie for movie in data_reader]
  rows = [[i]+rows[i][0:2]+[float(rows[i][2])]+rows[i][3:5]+rows[i][7:9]+[float(rows[i][9])]+[float(rows[i][10])]+rows[i][11:] for i in range(1,len(rows))] # don't need crew or original title

cur.executemany("INSERT INTO imdb_movies VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
con.commit()

# Fetch and clean up the description data
both = [row for row in cur.execute("SELECT rating, description, genre, budget, revenue, status, title, language, country, release_date FROM imdb_movies")]
both.sort() # sort by rating

ratings = [pair[0] for pair in both] # estimate as N(7,0.5)
genresBad = [pair[2] for pair in both]
budgets = [pair[3] for pair in both]
revenues = [pair[4] for pair in both]
descriptionsBad = [pair[1].lower().split(" ") for pair in both]
statusBad = [pair[5] for pair in both]
titleBad = [pair[6] for pair in both]
languageBad = [pair[7] for pair in both]
countryBad = [pair[8] for pair in both]
releases = [int(pair[9].replace('/', '')) for pair in both]

# list of uninformative words (propositions, articles, pronouns, etc)
props = ["by","&","it","and","but","or","","who","what","-","--","when","that","are","has","them","must","his","her","their","she","he","they","and","is","a","an","the","about","like","above","near","across","of","after","off","against","on","along","onto","among","opposite","around","out","as","outside","at","over","before","past","behind","round","below","since","beneath","than","beside","through","between","to","beyond","towards","by","under","despite","underneath","down","unlike","during","until","except","up","for","upon","from","via","in","with","inside","within","into","without"] #list of propositions
descriptionsC = []
descriptionsS = []

# Get rid of useless words
for d in descriptionsBad:
    newD = []
    newW = ""
    for w in d:
        if w in props:
            continue
        else:
            newD.append(w)
            newW += w
    descriptionsC.append(newD)
    descriptionsS.append(newW)

descScores = [scrabbleScore(d) for d in descriptionsS]
genreScores = [scrabbleScore(g) for g in genresBad]
titles = [scrabbleScore(t) for t in titleBad]
languages = [scrabbleScore(l) for l in languageBad]
countries = [scrabbleScore(c) for c in countryBad]
earningRatio = [revenues[i]/budgets[i] for i in range(len(revenues))]
status = []
for s in statusBad:
    if "Released" in s:
        status.append(0)
    elif "Post Production" in s:
        status.append(1)
    elif "In Production" in s:
        status.append(2)
    else:
        pass

# Get most common word and its Scrabble Score
#words = [Counter(d).most_common(1)[0][0] for d in descriptionsC]
#wordScores = [scrabbleScore(w) for w in words]

# Graph that ish!!
#plt.bar(words,ratings)
#plt.show()

# Get a linear regression model using wordScores to get rating

# input: training = training data, fold = which fold is beign set aside as test data, degree = class of model
# output = theta MLE, sigma^2 MLE
def MLE(training,fold,degree):
    score = []
    rating = []
    for i in range(len(training)):
        if i%5==fold:
            continue
        else:
            score.append(training[i][0])
            rating.append(training[i][1])
    n = len(score)
    phi = [] # get feature vector based on model class, going to be degree*n matrix
    for i in range(n):
        feature = [] # goin to be array of len(degree+1), ex. [x^3 x^2 x^1 x^0]
        for d in range(degree,-1,-1):
            if type(score[i]) == list:
                score[i] = [s**d for s in score[i]]
                feature += [score[i]]
            else:
                feature += [score[i]**d]
        phi += [numpy.transpose(feature)]
    # Plugging in values from Lecture 13
    # theta = (phi^T*phi)^-1*(phi^T*Y)
    theta = numpy.matmul(numpy.linalg.inv((numpy.matmul(numpy.transpose(phi),phi))),numpy.matmul(numpy.transpose(phi),rating))
    # sigma^2 = 1/n*(Y-phi*theta)^T*(Y-phi*theta)
    sigma = (1/n)*(numpy.matmul(numpy.transpose(numpy.subtract(rating,numpy.matmul(phi,theta))),numpy.subtract(rating,numpy.matmul(phi,theta))))
    return theta,sigma

# input: training = training data, fold = which fold to use as test data, line = MLE model equation
# output = mean squared error of specific model
def MSE(training,fold,line):
    total = 0
    n=0
    for i in range(len(training)):
        if i%5==fold:
            n+=1
            total += (training[i][1]-line(training[i][0]))**2
    return total/n


# TRY WITH DESCRIPTIONS
training = [[descScores[i],ratings[i]] for i in range(len(descScores))]

mle = MLE(training,0,1)
m = mle[0][0]
b = mle[0][1]
sigma = mle[1]

print("y=", m, "x+", b, "with variance ", sigma)

def line1(x): return m*x + b
mse = MSE(training,0,line1)
print("Mean Squared Error = ", mse)

plt.scatter(descScores,ratings, zorder=1) 
x = numpy.linspace(0, max(descScores), 100)
y = m*x + b
plt.plot(x, y, color="red", zorder=2)
plt.show()


mle2 = MLE(training,1,2)
m1 = mle2[0][0]
m2 = mle2[0][1]
b2 = mle2[0][2]
sigma2 = mle2[1]

print("y=", m1, "x^2+", m2, "x+", b2, "with variance ", sigma2)

def line2(x): return m1*x**2 + m2*x + b2
mse = MSE(training,1,line2)
print("Mean Squared Error = ", mse)

plt.scatter(descScores,ratings, zorder=1) 
x = numpy.linspace(0, max(descScores), 100)
y = m1*x**2 + m2*x + b2
plt.plot(x, y,color="red",zorder=2)
plt.show()

mle3 = MLE(training,2,3)
m1 = mle3[0][0]
m2 = mle3[0][1]
m3 = mle3[0][2]
b3 = mle3[0][2]
sigma3 = mle3[1]

print("y=", m1, "x^3+", m2, "x^2+", m3, "x+", b3, "with variance ", sigma3)

def line3(x): return m1*x**3 + m2*x**2 + m3*x + b3
mse = MSE(training,2,line3)
print("Mean Squared Error = ", mse)

plt.scatter(descScores,ratings, zorder=1) 
x = numpy.linspace(0, max(descScores), 100)
y = m1*x**3 + m2*x**2 + m3*x + b3
plt.plot(x, y,color="red",zorder=2)
plt.show()

# TRY WITH REVENUE/BUDGET
earnTrainingBad = [[earningRatio[i],ratings[i]] for i in range(len(descScores))]
earnTraining = []
earningRatioFix = []
ratingsFix = []
for i in range(len(earnTrainingBad)):
    if earnTrainingBad[i][0] < 100000:
        earnTraining.append(earnTrainingBad[i])
        earningRatioFix.append(earnTrainingBad[i][0])
        ratingsFix.append(earnTrainingBad[i][1])


Emle = MLE(earnTraining,3,1)
mE = Emle[0][0]
bE = Emle[0][1]
sigmaE = Emle[1]

print("y=", mE, "x+", bE, "with variance ", sigmaE)

def line4(x): return mE*x + bE
mse = MSE(earnTraining,3,line4)
print("Mean Squared Error = ", mse)

plt.scatter(earningRatioFix,ratingsFix, zorder=1) 
x = numpy.linspace(0, 100000, 100)
y = mE*x + bE
plt.plot(x, y, color="red",zorder=2)
plt.show()


Emle = MLE(earnTraining,4,2)
mE1 = Emle[0][0]
mE2 = Emle[0][1]
bE = Emle[0][2]
sigmaE = Emle[1]

print("y=", mE1, "x^2+", mE2, "x+", bE, "with variance ", sigmaE)

def line5(x): return mE1*x**2 + mE2*x + bE
mse = MSE(earnTraining,4,line5)
print("Mean Squared Error = ", mse)

plt.scatter(earningRatioFix,ratingsFix, zorder=1) 
x = numpy.linspace(0, 100000, 100)
y = mE1*x**2 + mE2*x + bE
plt.plot(x, y, color="red",zorder=2)
plt.show()

Emle = MLE(earnTraining,0,3)
mE1 = Emle[0][0]
mE2 = Emle[0][1]
mE3 = Emle[0][2]
bE = Emle[0][3]
sigmaE = Emle[1]

print("y=", mE1, "x^3+", mE2, "x^2+", mE3, "x+", bE, "with variance ", sigmaE)

def line6(x): return mE1*x**3 + mE2*x**2+ mE3*x + bE
mse = MSE(earnTraining,0,line6)
print("Mean Squared Error = ", mse)

plt.scatter(earningRatioFix,ratingsFix, zorder=1) 
x = numpy.linspace(0, 100000, 100)
y = mE1*x**3 + mE2*x**2+ mE3*x + bE
plt.plot(x, y, color="red",zorder=2)
plt.show()

# TRY WITH STATUS
multTraining = [[status[i],ratings[i]] for i in range(len(descScores))]

mMle = MLE(multTraining,0,1)
mM = mMle[0][0]
bM = mMle[0][1]
sigmaM = mMle[1]

print("y=", mM, "x+", bM, "with variance ", sigmaM)

def line7(x): return mM*x + bM
mse = MSE(multTraining,0,line7)
print("Mean Squared Error = ", mse)

plt.scatter(status,ratings, zorder=1) 
x = numpy.linspace(-1, 5, 100)
y = mM*x + bM
plt.plot(x, y, color="red",zorder=2)
plt.show()

mMle = MLE(multTraining,1,2)
mM1 = mMle[0][0]
mM2 = mMle[0][1]
bM = mMle[0][2]
sigmaM = mMle[1]

print("y=", mM1, "x^2+", mM2, "x+", bM, "with variance ", sigmaM)

def line8(x): return mM1*x**2 + mM2*x + bM
mse = MSE(multTraining,1,line8)
print("Mean Squared Error = ", mse)

plt.scatter(status,ratings, zorder=1) 
x = numpy.linspace(-1, 5, 100)
y = mM1*x**2 + mM2*x + bM
plt.plot(x, y, color="red",zorder=2)
plt.show()

mMle = MLE(multTraining,2,3)
mM1 = mMle[0][0]
mM2 = mMle[0][1]
mM3 = mMle[0][2]
bM = mMle[0][3]
sigmaM = mMle[1]

print("y=", mM1, "x^3+", mM2, "x^2+", mM3, "x+", bM, "with variance ", sigmaM)

def line9(x): return mM1*x**3 + mM2*x**2 + mM3*x + bM
mse = MSE(multTraining,2,line9)
print("Mean Squared Error = ", mse)

plt.scatter(status,ratings, zorder=1) 
x = numpy.linspace(-1, 5, 100)
y = mM1*x**3 + mM2*x**2 + mM3*x + bM
plt.plot(x, y, color="red",zorder=2)
plt.show()

# ALL FEATURES
allInputsTot = [[descScores[i],genreScores[i],status[i],budgets[i],revenues[i],titles[i],languages[i],countries[i],releases[i]] for i in range(len(revenues))]
allInputsTrain = []
allInputsTest = []
ratingsTrain = []
ratingsTest = []
k = 4
for i in range(len(revenues)):
    if i%5 == k:
        allInputsTest.append(allInputsTot[i])
        ratingsTest.append(ratings[i])
    else:
        allInputsTrain.append(allInputsTot[i])
        ratingsTrain.append(ratings[i])
clf = linear_model.LinearRegression()
clf.fit(allInputsTrain,ratingsTrain)
pred = clf.predict(allInputsTest)
mse = sum([(ratingsTest[i]-pred[i])**2 for i in range(len(ratingsTest))])
print("MSE = ", mse/len(ratingsTest))
print(clf.coef_)
print(clf.intercept_)
exit(0)
clf.coef_ = list(clf.coef_)
print(clf.coef_)
x = numpy.linspace(-1, 1000, 100)
y = 2.8e-04*x**8 + 1.9e-01*x**7 + -2.9e+01*x**6 + -1.1e-07*x**5 + 1.9e-08*x**4 + 2.0e-02*x**3 + 4.7e-01*x**2 + 1.4e-07
plt.plot(x,y)
plt.show()

########

# CODE FOR GETTING OTHER DATABASE DATA (may need later)

########


#plt.scatter(descScores,ratings) 
#plt.show() 

#cur.execute("CREATE TABLE movies(id, title, release_date, rating, description, popularity)")

#with open('movies.CSV', newline='', errors='ignore') as csvfile:
#  data_reader = csv.reader(csvfile)
#  rows = [movie for movie in data_reader]
#  rows = [rows[i][1:3]+rows[i][4:5]+[float(rows[i][6])]+rows[i][3:4]+[float(rows[i][5])] for i in range(1,len(rows))] # don't need vote count

#cur.executemany("INSERT INTO movies VALUES(?, ?, ?, ?, ?, ?)", rows)
#con.commit()


#cur.execute("CREATE TABLE metadata(id, title, release_date, rating, genre, description, popularity, status, language, budget, revenue, country, runtime)")

#with open('movies_metadata.CSV', newline='', errors='ignore') as csvfile:
#  data_reader = csv.reader(csvfile)
#  rows = [movie for movie in data_reader]
#  for row in rows:
#    if len(row[1]) != 0:
#      row[1] = True # turn into bool
#    else:
#      row[1] = False
#    row[3] = row[3][0]["name"] # just keep first genre
#    row[13] = row[13][0]["name"] # just keep first country


#print(rows[0])
#cur.executemany("INSERT INTO movies VALUES(?, ?, ?, ?, ?, ?)", rows)
#con.commit()