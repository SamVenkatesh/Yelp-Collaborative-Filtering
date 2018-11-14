from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
import pyspark

if __name__ == '__main__':
    spark = SparkSession\
        .builder\
        .appName("ItemBasedCF")\
        .getOrCreate()

    my_schema = StructType([StructField("user_id", StringType(), True),
                            StructField("business_id", StringType(), True),
                            StructField("stars", DoubleType(), True)])

    req_cols = ['user_id', 'business_id', 'stars']

    ratings = spark.read.csv('code/data/train_review.csv', header=True, schema=my_schema)
    ratings = ratings.select(req_cols)

    test_ratings = spark.read.csv('code/data/test_review.csv', header=True, schema=my_schema)
    test_ratings = ratings.select(req_cols)

    user_indexer = StringIndexer(inputCol = 'user_id', outputCol = 'user_idx', handleInvalid = 'skip')
    business_indexer = StringIndexer(inputCol = 'business_id', outputCol = 'business_idx', handleInvalid = 'skip')

    user_indexed_ratings_training = user_indexer.fit(ratings).transform(ratings)
    total_indexed_training = business_indexer.fit(user_indexed_ratings_training).transform(user_indexed_ratings_training)

    user_indexed_ratings_test = user_indexer.fit(test_ratings).transform(test_ratings)
    total_indexed_test = business_indexer.fit(user_indexed_ratings_test).transform(user_indexed_ratings_test)
    
    training_indexed = total_indexed_training.na.drop()
    test_indexed = total_indexed_test.na.drop()

    print '=================================================================================='
    print 'TRAINING DATA:'
    print '=================================================================================='
    training_indexed.show()
    print '=================================================================================='
    print 'TEST DATA:'
    print '=================================================================================='
    test_indexed.show()
    print '=================================================================================='

    als = ALS(maxIter=5, regParam=0.01, userCol="user_idx", itemCol="business_idx", 
            ratingCol="stars", coldStartStrategy="drop", nonnegative=True)

    model = als.fit(training_indexed)

    predictions = model.transform(test_indexed)
    # TODO: Fix predictions exceeding max rating 5
    predictions.show()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))