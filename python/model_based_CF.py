#import findspark
#findspark.init()
import time
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
    global d
    if (value.isnumeric()):
        return value
    if(value not in d):
        d[value] = c
        k = c
        c += 1
        return k
    else:
        return d[value]


time_start = time.time()

data = sc.textFile("/Users/shreyabhat/Documents/GitHub/Yelp-Collaborative-Filtering/data/train_review.csv")

ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(hash_string(l[2]))))

rank = 15
numIterations = 20
model = ALS.train(ratings, rank, numIterations)
test_data = sc.textFile("/Users/shreyabhat/Documents/GitHub/Yelp-Collaborative-Filtering/data/test_review.csv")

test_ratings = test_data.map(lambda l: l.split(',')).map(lambda l: Rating(hash_string(l[0]), hash_string(l[1]), float(hash_string(l[2]))))
tr = test_ratings.map(lambda r: ((r[0], r[1]), float(r[2])))
testdata = test_ratings.map(lambda p: (p[0], p[1]))

predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), float(r[2])))
#print(predictions.map(lambda t : t[0][1]).mean())
#print(tr.map(lambda  t : t[0][1]).mean())

ratesAndPreds = tr.join(predictions)

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()**0.5
print("Mean Squared Error = " + str(MSE))
time_end = time.time()

print("Total time = "+str(time_end - time_start))

