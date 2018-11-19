import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
sc = SparkContext.getOrCreate()

global d
global c
d = dict()
c = 0
def toCSVLine(data):
  return ','.join(str(d) for d in data)

def hash_string(value):
    global c
    if(value not in d):
        d[value] = c
        k = c
        c += 1
        return k
    else:
        return d[value]




data = sc.textFile("/Users/shreyabhat/Documents/GitHub/Yelp-Collaborative-Filtering/data/train_review.csv")

ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(l[2])))

rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)
test_data = sc.textFile("/Users/shreyabhat/Documents/GitHub/Yelp-Collaborative-Filtering/data/test_review.csv")
test_ratings = test_data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(l[2])))
tr = test_ratings.map(lambda r: ((r[0], r[1]), float(r[2])))
testdata = test_ratings.map(lambda p: (p[0], p[1]))

predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), float(r[2])))
print(predictions.map(lambda t : t[0][1]).mean())
print(tr.map(lambda  t : t[0][1]).mean())

ratesAndPreds = tr.join(predictions)

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

