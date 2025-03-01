import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
import json
from pyspark.sql import functions as F
from datetime import datetime

# Create Spark session
spark = SparkSession.builder \
    .appName("SentimentAnalysisTest") \
    .getOrCreate()

# Load models
feature_model = PipelineModel.load("model/feature_pipeline_model")
lr_model = LogisticRegressionModel.load("model/logistic_regression_model")

# Load new data
new_data = spark.read.csv("test/data.csv", header=True, inferSchema=True)

# Add date if not present
if "Date" not in new_data.columns:
    new_data = new_data.withColumn("Date", F.lit(datetime.now().strftime('%Y-%m-%d')))

# Process features
processed_data = feature_model.transform(new_data)

# Make predictions
predictions = lr_model.transform(processed_data)

# Load label mapping
try:
    with open("label_to_index_mapping.json", "r") as file:
        label_mapping = json.load(file)
except FileNotFoundError:
    print("label_to_index_mapping.json not found, using default mapping.")
    label_mapping = {
        "label 0": "Negative",
        "label 1": "Neutral",
        "label 2": "Positive"
    }

# Create index to label mapping
index_to_label = {int(key.split()[1]): label for key, label in label_mapping.items()}

# Add emotion labels
predictions_with_emotions = predictions.withColumn(
    "Emotion",
    F.when(F.col("prediction") == 0, index_to_label.get(0, "Unknown"))
    .when(F.col("prediction") == 1, index_to_label.get(1, "Unknown"))
    .when(F.col("prediction") == 2, index_to_label.get(2, "Unknown"))
    .otherwise("Unknown")
)

# Select output columns
output_columns = ["Date", "Cleaned Tweet Content", "prediction", "Emotion"]

# Show predictions
predictions_with_emotions.select(output_columns).show(5)

# Save results
output_path = "test_results"
predictions_with_emotions.select(output_columns).write.csv(
    output_path,
    header=True,
    mode="overwrite"
)

spark.stop()
