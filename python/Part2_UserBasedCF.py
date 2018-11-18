import sys
from pyspark import SparkContext, SparkConf
import math
import time
from collections import defaultdict

def convert(train_file, test_file):
    train_data = []
    all_users, all_business = set(), set()
    users_to_business_train, users_to_business_test, test_data = [], [], []

    with open(train_file) as f:
        for line in f.readlines()[1:]:
            parts = line.split(",")
            train_data.append((parts[0], parts[1], float(parts[2])))

    with open(test_file) as f:
        for line in f.readlines()[1:]:
            parts = line.split(",")
            test_data.append((parts[0], parts[1], float(parts[2])))

    return train_data, test_data



def correlation(prod_u1, prod_u2):
	intersection = set([p for p,r in prod_u1]).intersection(set([p for p,r in prod_u2]))
	if len(intersection) == 0:
		return 0.0,0.0
	prod1 = [r for prod,r in prod_u1 if prod in intersection]
	prod2 = [r for prod,r in prod_u2 if prod in intersection]
	prod1_mean, prod2_mean = sum(prod1) / len(prod1), sum(prod2) / len(prod2)

	prod1 = [prod-prod1_mean for prod in prod1]
	prod2 = [prod - prod2_mean for prod in prod2]
	res = 0.0
	for i in range(len(prod1)):
		res += prod1[i] * prod2[i]

	if res == 0:
		return 0.0, prod2_mean

	n1 = math.sqrt(sum(list(map(lambda x:x*x, prod1))))
	n2 = math.sqrt(sum(list(map(lambda x: x * x, prod2))))
	res = res/(n1*n2)
	return res, prod2_mean



def pearson(user, product, userProduct, productUser, avgUser, avgProduct):
	if user not in userProduct or product not in productUser:
		return ((user,product), 3.5)

	user_list = [(u,r) for u,r in productUser[product] if u != user]

	s1, s2 = 0.0, 0.0
	for u, r in user_list:
		corr = correlation(userProduct[user], userProduct[u])
		if corr[0] >= 0:
			s1 += (r - corr[1]) * corr[0]
			s2 += abs(corr[0])

	if s2 == 0:
		pred = (avgUser[user] + avgProduct[product]) / 2
		return ((user,product),pred)
	else:
		pred = avgUser[user] + (s1 / s2)
		return ((user,product),pred)



start = time.time()
conf = SparkConf().setMaster('local[*]').setAppName('task2')
sc = SparkContext(conf=conf)

log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger(__name__)

fileName1 = sys.argv[1]
fileName2 = sys.argv[2]

train_data, test_data = convert(fileName1, fileName2)

rdd1 = sc.parallelize(train_data)
rdd2 = sc.parallelize(test_data)


test_rdd = rdd2.map(lambda l: ((l[0], l[1]), 1))

productUserRating = rdd1.map(lambda l:(l[1], (l[0],float(l[2])))).groupByKey().collectAsMap()
userProductRating = rdd1.map(lambda l:(l[0], (l[1],float(l[2])))).groupByKey().collectAsMap()

avgUser = defaultdict(float)
for u in userProductRating:
	ratings = [r for p,r in userProductRating[u]]
	avgUser[u] = sum(ratings) / len(ratings)

avgProduct = defaultdict(float)
for p in productUserRating:
	ratings = [r for u,r in productUserRating[p]]
	avgProduct[p] = sum(ratings) / len(ratings)

final_ratings = test_rdd.map(lambda row: pearson(row[0][0], row[0][1], userProductRating, productUserRating,avgUser, avgProduct))


ratesAndPreds = rdd2.map(lambda r: ((r[0], r[1]), r[2])).join(final_ratings)
ratesAndPreds = ratesAndPreds.map(lambda l: (l[0], abs(l[1][0] - l[1][1])))

one = ratesAndPreds.filter(lambda l: l[1] >= 0.0 and l[1] < 1.0).count()
two = ratesAndPreds.filter(lambda l: l[1] >= 1.0 and l[1] < 2.0).count()
three = ratesAndPreds.filter(lambda l: l[1] >= 2.0 and l[1] < 3.0).count()
four = ratesAndPreds.filter(lambda l: l[1] >= 3.0 and l[1] < 4.0).count()
five = ratesAndPreds.filter(lambda l: l[1] >= 4.0).count()


MSE = math.sqrt(ratesAndPreds.map(lambda r: r[1]**2).mean())
predictions_dict = final_ratings.collectAsMap()

print(">=0 and <1: ", one)
print(">=1 and <2: ", two)
print(">=2 and <3: ", three)
print(">=3 and <4: ", four)
print(">=4: ", five)
print("RMSE: " + str(MSE))

# writeToFile(final_ratings.collect())
print("Time taken: ", time.time() - start)
with open("Output_UserBasedCF.txt", "w") as f:
	for u, b in sorted(predictions_dict, key=lambda k,: (k[0], k[1])):
		f.write(u + ", " + b + ", " + str(predictions_dict[(u, b)]) + "\n")

