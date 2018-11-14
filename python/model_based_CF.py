import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
sc = SparkContext.getOrCreate()
import numpy as np
#TOTAL = 1000000
#dots = sc.parallelize([2.0 * np.random.random(2) - 1.0 for i in range(TOTAL)]).cache()
#print("Number of random points:", dots.count())
#stats = dots.stats()
#print('Mean:', stats.mean())
#print('stdev:', stats.stdev())
def toCSVLine(data):
  return ','.join(str(d) for d in data)
def hash_string(value):
    score = 0
    depth = 1
    for char in value:
        score += (ord(char)) * depth
        depth /= 256.
    return score
data = sc.textFile("data/train_review.csv")
data.collect()
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(l[2])))
rank = 1
numIterations = 1
model = ALS.train(ratings, rank, numIterations)
test_data = sc.textFile("data/test_review.csv")
test_ratings = test_data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(l[2])))

testdata = test_ratings.map(lambda p: (p[0], p[1]))
l1 = []
l2 = []
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
l1.append(predictions.map(lambda p : p[1].value()))
ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)


MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

