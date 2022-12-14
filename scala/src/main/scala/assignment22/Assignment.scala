package assignment22

import scala.math._
import org.apache.spark.ml.feature.{MinMaxScaler, Normalizer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, mean, struct, sum, when}
import org.apache.spark.sql.Dataset


class Assignment {

  val spark: SparkSession = SparkSession.builder()
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()
  // suppress informational log messages related to the inner working of Spark
  spark.sparkContext.setLogLevel("ERROR")

  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark.read
    .option(key = "header", value = true)
    .option("inferSchema", "true")
    .csv("data/dataD2.csv")

  // the data frame to be used in task 2
  val dataD3: DataFrame = spark.read
    .option(key = "header", value = true)
    .option("inferSchema", "true")
    .csv("data/dataD3.csv")

  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  // Fatal will be 0 and Ok will be 1
  //***********************************************************************************
  val dataD2WithLabels: DataFrame = dataD2.withColumn("label", when(col("label") === "Fatal", 0)
    .when(col("label") === "Ok", 1))


  //Task 1 is done by me
  //This function finds certain amount of cluster centers corresponding to parameters k value
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    //create vectorAssembler
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("featuresUnscaled")

    //creating scaler to scale to data that this function handles
    val scaler = new MinMaxScaler()
      .setInputCol("featuresUnscaled")
      .setOutputCol("features")

    //creating the pipeline
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))

    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)

    //creating new kmeans instance according to the parameter k
    val kmeans = new KMeans()
      .setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedData)
    var centers = new Array[(Double, Double)](k)

    //mapping the cluster centers to array that the function returns
    centers = kmModel.clusterCenters.map(center => (center(0), center(1)))

    //the cluster centers
    centers.foreach(println)
    centers
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    //create vectorAssembler
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("featuresUnscaled")

    val scaler = new MinMaxScaler()
      .setInputCol("featuresUnscaled")
      .setOutputCol("features")

    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))

    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)

    //creating new kmeans instance according to the parameter k
    val kmeans = new KMeans()
      .setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedData)
    var centers = new Array[(Double, Double, Double)](k)

    centers = kmModel.clusterCenters.map(center => (center(0), center(1), center(2)))
    //the cluster centers
    centers.foreach(println)
    centers
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    //create vectorAssembler
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "label"))
      .setOutputCol("featuresUnscaled")

    val scaler = new MinMaxScaler()
      .setInputCol("featuresUnscaled")
      .setOutputCol("features")

    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))

    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)

    //creating new kmeans instance according to the parameter k
    val kmeans = new KMeans()
      .setK(k).setSeed(1L)

    val kmModel = kmeans.fit(transformedData)
    val kmTransform = kmModel.transform(transformedData)
    val resultsRaw = kmTransform.groupBy("prediction").agg(mean("a"), mean("b"), sum("label")).sort("sum(label)").take(2)
    var centers = new Array[(Double, Double)](2)

    centers = resultsRaw.map(result => (result(1).toString.toDouble, result(2).toString.toDouble))

    //the cluster centers
    centers.foreach(println)
    centers
  }

  // Parameter low is the lowest k and high is the highest one.
  //Task 4 is done by me. 
  //The function calculates silhouette scores which calculate how good clustering is
  //Function returns Array with pair that has cluster center amount and silhouette score for that
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {

    var k_score_pairs = new Array[(Int, Double)](high - low + 1)

    //helper variable for the .map() below
    var temp = new Array[(Int, Double)](high - low + 1)

    //create vectorAssembler
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("featuresUnscaled")

    val scaler = new MinMaxScaler()
      .setInputCol("featuresUnscaled")
      .setOutputCol("features")

    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))

    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)

    var a = low

    //mapping through every different cluster center amounts 
    //calculate silhouette score for that amount and put them to array that will be returned
    temp.indices.map(index => {
      val kmeans = new KMeans()
        .setK(a).setSeed(1L)

      val kmModel = kmeans.fit(transformedData)

      val predictions = kmModel.transform(transformedData)

      val evaluator = new ClusteringEvaluator()

      //calculating the silhouette score for the clusters
      val silhouette = evaluator.evaluate(predictions)

      k_score_pairs(index) = (a, silhouette)
      a += 1
    })

      k_score_pairs
  }

}
