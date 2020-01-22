# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import *
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SQLContext
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import countDistinct
from som.batch_som import SOM
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# begin SparkSession
conf = SparkConf().setMaster("local[*]")
sc = SparkContext.getOrCreate(conf)
spark = SparkSession(sc)

# step 1 load data from hdfs
def load_data(data):
    input_data = spark.read.csv(data, header=False, inferSchema=True, mode="DROPMALFORMED", encoding='UTF-8')
    #return input_data.show(n=3,truncate=False)
    return input_data
    
# step 2 shuffling data # repartitioning to manage level of parallelism
def shuffling(input_data, k):
    input_data = input_data.repartition(k)
    #input_data.show(3)
    return input_data

# step 3  selecting features from dataframe
def assembler(input_data,inputCols):
    # inputCols = input_data.columns[0:] 
    assembler = VectorAssembler(inputCols = inputCols,outputCol = "raw_features")
    input_features = assembler.transform(input_data)
    #input_features.show(3)
    return input_features

# step 4 MaxAbsScaler is used to preserve the original features and distance as we will later use Euclidean distance
def scaler(input_features):
    scaler = MaxAbsScaler(inputCol="raw_features", outputCol="features")
    scalerModel = scaler.fit(input_features)
    scaledData = scalerModel.transform(input_features).drop("raw_features")
    #scaledData.show(3)
    return scaledData
	
# step 6 cluster for scaledData 
def cluster(data, n_clusters):
    model = KMeans().setK(n_clusters).setSeed(1).setFeaturesCol("features").fit(scaledData)
    centers = model.clusterCenters()
    #print("Cluster Centers: ")
    #for center in centers:
     #   print(center)
    cl_labels = model.transform(scaledData).select('prediction')
    gr = cl_labels.groupBy("prediction").agg(countDistinct("prediction"))
    #gr.show()
    return cl_labels, gr.show()

def calculate_map_size(input_data):
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        import numpy as np
        dlen = input_data.count()
        dim = len(input_data.dtypes)
        #D = df
        #dlen = D.shape[0]
        #dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf
      
        size1 = min(munits, round(np.sqrt(munits / np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]