# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

!pip install pyspark pandas

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel

spark = SparkSession.builder \
    .appName("PropertyPricePrediction") \
    .config("spark.sql.shuffle.partitions", "800") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

file_path = '/content/drive/MyDrive/202304.csv'

df = spark.read.csv(file_path, header=True, inferSchema=True)

column_names =['Transaction_ID', 'price', 'Date_of_Transfer',
'postcode', 'Property_Type', 'Old/New',
'Duration', 'PAON', 'SAON',
'Street', 'Locality', 'Town/City',
'District', 'County', 'PPDCategory_Type',
'Record_Status - monthly_file_only'
]

df = df.toDF(*column_names)

# df.show(100)

red_col=('Transaction_ID','PAON', 'SAON',
'Street', 'Locality', 'PPDCategory_Type',
'Record_Status - monthly_file_only'
)

df=df.drop(*red_col)
df = df.na.drop()
df=df.dropDuplicates()
# df.show(100)

from pyspark.sql.functions import year, month, dayofmonth, to_date

df = df.withColumn('Date_of_Transfer', to_date(df['Date_of_Transfer'], 'yyyy-MM-dd'))
df = df.withColumn('Year', year(df['Date_of_Transfer']))
df = df.withColumn('Month', month(df['Date_of_Transfer']))
df = df.withColumn('Day', dayofmonth(df['Date_of_Transfer']))

fraction = 0.00001
df_sampled = df.sample(fraction=fraction, seed=42)


df_sampled = df_sampled.dropDuplicates()

df_sampled = df_sampled.na.drop()

df_sampled.write.csv('/content/cleaned_sampled_file.csv', header=True)

df_sampled.write.parquet('/content/cleaned_sampled_file.parquet')

df_sampled.show()

file_path='/content/cleaned_sampled_file.csv'
dff=spark.read.csv(file_path)
print(dff.count())

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


df_sampled = df_sampled.select('price', 'Property_Type', 'Town/City', 'Old/New', 'postcode').dropna()
df_sampled = df_sampled.withColumn("Property_Type", col("Property_Type").cast("string"))
df_sampled = df_sampled.withColumn("Town/City", col("Town/City").cast("string"))
df_sampled = df_sampled.withColumn("Old/New", col("Old/New").cast("string"))
df_sampled = df_sampled.withColumn("price", col("price").cast("int"))

indexer_prop_type = StringIndexer(inputCol="Property_Type", outputCol="Property_Type_index")
indexer_city = StringIndexer(inputCol="Town/City", outputCol="Town/City_index")
indexer_old_new = StringIndexer(inputCol="Old/New", outputCol="Old/New_index")
indexer_postcode = StringIndexer(inputCol="postcode", outputCol="postcode_index")

encoded_prop_type = OneHotEncoder(inputCol="Property_Type_index", outputCol="Property_Type_encoded")
encoded_city = OneHotEncoder(inputCol="Town/City_index", outputCol="Town/City_encoded")
encoded_old_new = OneHotEncoder(inputCol="Old/New_index", outputCol="Old/New_encoded")
encoded_postcode = OneHotEncoder(inputCol="postcode_index", outputCol="postcode_encoded")

df_sampled = indexer_prop_type.fit(df_sampled).transform(df_sampled)
df_sampled = indexer_city.fit(df_sampled).transform(df_sampled)
df_sampled = indexer_old_new.fit(df_sampled).transform(df_sampled)
df_sampled = indexer_postcode.fit(df_sampled).transform(df_sampled)

df_sampled = encoded_prop_type.fit(df_sampled).transform(df_sampled)
df_sampled = encoded_city.fit(df_sampled).transform(df_sampled)
df_sampled = encoded_old_new.fit(df_sampled).transform(df_sampled)
df_sampled = encoded_postcode.fit(df_sampled).transform(df_sampled)

assembler = VectorAssembler(
    inputCols=["Property_Type_encoded", "Town/City_encoded", "Old/New_encoded", "postcode_encoded"],
    outputCol="features"
)
df = assembler.transform(df_sampled)

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(train_df)

predictions = lr_model.transform(test_df)

evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

spark.stop()
