from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json

# Configurer Spark
conf = SparkConf() \
    .setAppName("SentimentAnalysis") \
    .set("spark.python.worker.reuse", "true") \
    .set("spark.sql.broadcastTimeout", "1200") \
    .set("spark.executor.memory", "2g") \
    .set("spark.driver.memory", "2g")

# Créer la session Spark
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Charger les données
try:
    print("Chargement des données...")
    data = spark.read.csv("train/data.csv", header=True, inferSchema=True)
    print("Données chargées avec succès.")
    data.show(5)
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    spark.stop()
    exit()

# Vérifier si les colonnes nécessaires sont présentes
if "Cleaned Tweet Content" not in data.columns or "Sentiment" not in data.columns:
    print("Les colonnes nécessaires ne sont pas présentes dans les données.")
    spark.stop()
    exit()

# Pipeline de traitement des caractéristiques
feature_pipeline = Pipeline(stages=[
    Tokenizer(inputCol="Cleaned Tweet Content", outputCol="words"),
    StopWordsRemover(inputCol="words", outputCol="filtered_words"),
    HashingTF(inputCol="filtered_words", outputCol="raw_features"),
    IDF(inputCol="raw_features", outputCol="features")
])

# Adapter et enregistrer le pipeline des caractéristiques
try:
    print("Construction du pipeline de traitement des caractéristiques...")
    feature_model = feature_pipeline.fit(data)
    feature_model.write().overwrite().save("model/feature_pipeline_model")  # Enregistrer le pipeline
    processed_data = feature_model.transform(data)
    print("Pipeline des caractéristiques construit et enregistré avec succès.")
except Exception as e:
    print(f"Erreur dans le pipeline des caractéristiques : {e}")
    spark.stop()
    exit()

# Indexation des labels
indexer = StringIndexer(inputCol="Sentiment", outputCol="label", handleInvalid="skip")
try:
    print("Indexation des labels...")
    fitted_indexer = indexer.fit(data)
    indexed_data = fitted_indexer.transform(processed_data)
    print("Indexation des labels terminée.")
except Exception as e:
    print(f"Erreur dans l'indexation des labels : {e}")
    spark.stop()
    exit()

# Sauvegarder la correspondance des labels
label_mapping = {f"label {i}": label for i, label in enumerate(fitted_indexer.labels)}
with open("label_to_index_mapping.json", "w") as f:
    json.dump(label_mapping, f)

# Séparer les données en ensembles d'entraînement et de test
train_data, test_data = indexed_data.randomSplit([0.8, 0.2], seed=1234)

# Modèle de régression logistique
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Validation croisée et grille d'hyperparamètres
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.01])
             .addGrid(lr.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction"),
                    numFolds=3)

# Entraîner le modèle
try:
    print("Entraînement du modèle...")
    cvModel = cv.fit(train_data)
    print("Modèle entraîné avec succès.")
except Exception as e:
    print(f"Erreur lors de l'entraînement du modèle : {e}")
    spark.stop()
    exit()

# Évaluer le modèle
predictions = cvModel.bestModel.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Précision du modèle : {accuracy:.4f}")

# Afficher les métriques pour chaque label
predictionAndLabels = predictions.select("prediction", "label").rdd.map(tuple)
from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(predictionAndLabels)
for idx, label in enumerate(fitted_indexer.labels):
    numeric_label = float(idx)
    precision = metrics.precision(numeric_label)
    recall = metrics.recall(numeric_label)
    f1_score = metrics.fMeasure(numeric_label)

    print(f"Label '{label}':")
    print(f"  Précision: {precision}")
    print(f"  Rappel: {recall}")
    print(f"  Score F1: {f1_score}")
    print("---------------------------")

# Sauvegarder le meilleur modèle
cvModel.bestModel.write().overwrite().save("model/logistic_regression_model")

# Arrêter la session Spark
spark.stop()
