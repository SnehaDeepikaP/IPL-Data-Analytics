"""
ipl_pipeline.py  —  IPL PySpark Data Processing + ML Training
==============================================================
Run this ONCE locally to:
  1. Load & clean IPL.csv with PySpark
  2. Run EDA aggregations and save them as CSV/Parquet for the dashboard
  3. Train 4 ML models and save AUC results

Usage:
    python ipl_pipeline.py --data IPL.csv --out data/

Requirements:
    pip install pyspark pandas plotly
"""

import os
import time
import argparse
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier,
    GBTClassifier, DecisionTreeClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="IPL PySpark Pipeline")
parser.add_argument("--data", default="IPL.csv", help="Path to IPL.csv")
parser.add_argument("--out",  default="data",    help="Output directory for CSVs")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 1. Spark Session ──────────────────────────────────────────────────────────
print("Starting Spark...")
spark = (
    SparkSession.builder
    .appName("IPL_ML")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version: {spark.version}")

# ── 2. Load & cast ────────────────────────────────────────────────────────────
print(f"\nLoading data from: {args.data}")
sdf = spark.read.csv(args.data, header=True, inferSchema=True)

int_cols = [
    "runs_batter", "runs_extras", "runs_total", "balls_faced",
    "valid_ball", "year", "over", "ball", "runs_bowler",
]
for c in int_cols:
    if c in sdf.columns:
        sdf = sdf.withColumn(c, F.col(c).cast(IntegerType()))

print(f"Total rows loaded: {sdf.count():,}")
sdf.cache()

# ── helper ────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, name: str):
    path = os.path.join(args.out, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}  ({len(df)} rows)")


# ── 3. EDA aggregations ───────────────────────────────────────────────────────
print("\n[EDA] Running aggregations...")

# Season-wise total runs
save(
    sdf.groupBy("year")
       .agg(F.sum("runs_batter").alias("total_runs"))
       .orderBy("year")
       .toPandas(),
    "season_runs",
)

# Wicket kinds
save(
    sdf.filter(F.col("wicket_kind").isNotNull())
       .groupBy("wicket_kind")
       .count()
       .toPandas(),
    "wicket_kinds",
)

# Match results
save(
    sdf.dropDuplicates(["match_id"])
       .groupBy("win_outcome")
       .count()
       .toPandas(),
    "match_results",
)

# Toss decisions
save(
    sdf.dropDuplicates(["match_id"])
       .groupBy("toss_decision")
       .count()
       .toPandas(),
    "toss_decisions",
)

# Top batters
save(
    sdf.groupBy("batter")
       .agg(
           F.sum("runs_batter").alias("runs"),
           F.sum("valid_ball").alias("balls"),
       )
       .withColumn("SR", F.round(F.col("runs") / F.col("balls") * 100, 1))
       .orderBy(F.desc("runs"))
       .limit(50)
       .toPandas(),
    "top_batters",
)

# Runs by over
save(
    sdf.groupBy("over")
       .agg(F.sum("runs_batter").alias("runs_batter"))
       .orderBy("over")
       .toPandas(),
    "runs_by_over",
)

# Top bowlers (min 60 valid balls)
save(
    sdf.groupBy("bowler")
       .agg(
           F.sum(F.when(F.col("wicket_kind").isNotNull(), 1).otherwise(0)).alias("wickets"),
           F.sum("runs_bowler").alias("runs_given"),
           F.sum("valid_ball").alias("balls"),
       )
       .filter(F.col("balls") >= 60)
       .withColumn("Economy", F.round(F.col("runs_given") / (F.col("balls") / 6), 2))
       .orderBy(F.desc("wickets"))
       .limit(50)
       .toPandas(),
    "top_bowlers",
)

# Wickets by over
save(
    sdf.filter(F.col("wicket_kind").isNotNull())
       .groupBy("over")
       .agg(F.count("wicket_kind").alias("wicket_kind"))
       .orderBy("over")
       .toPandas(),
    "wickets_by_over",
)

# Team runs
save(
    sdf.groupBy("batting_team")
       .agg(
           F.sum("runs_batter").alias("runs"),
           F.countDistinct("match_id").alias("matches"),
       )
       .withColumn("RPM", F.round(F.col("runs") / F.col("matches"), 1))
       .orderBy(F.desc("runs"))
       .toPandas(),
    "team_runs",
)

# Team trends by year
save(
    sdf.groupBy("year", "batting_team")
       .agg(F.sum("runs_batter").alias("runs_batter"))
       .orderBy("year")
       .toPandas(),
    "team_trends",
)

# Season stats (sixes, fours, wickets, avg)
save(
    sdf.groupBy("year")
       .agg(
           F.sum("runs_batter").alias("total_runs"),
           F.sum(F.when(F.col("wicket_kind").isNotNull(), 1).otherwise(0)).alias("total_wickets"),
           F.countDistinct("match_id").alias("matches"),
           F.sum(F.when(F.col("runs_batter") == 6, 1).otherwise(0)).alias("sixes"),
           F.sum(F.when(F.col("runs_batter") == 4, 1).otherwise(0)).alias("fours"),
       )
       .withColumn("avg", F.round(F.col("total_runs") / F.col("matches"), 1))
       .orderBy("year")
       .toPandas(),
    "season_stats",
)

# Sixes and fours per season
save(
    sdf.groupBy("year")
       .agg(
           F.sum(F.when(F.col("runs_batter") == 6, 1).otherwise(0)).alias("sixes"),
           F.sum(F.when(F.col("runs_batter") == 4, 1).otherwise(0)).alias("fours"),
       )
       .orderBy("year")
       .toPandas(),
    "sixes_fours",
)

# Summary metrics (single-row)
matches  = sdf.select("match_id").distinct().count()
runs     = int(sdf.agg(F.sum("runs_batter")).collect()[0][0] or 0)
balls    = int(sdf.agg(F.sum("valid_ball")).collect()[0][0] or 0)
wickets  = int(sdf.filter(F.col("wicket_kind").isNotNull()).count())
avg_score = round(
    sdf.groupBy("match_id").agg(F.sum("runs_total").alias("s"))
       .agg(F.avg("s"))
       .collect()[0][0] or 0,
    1,
)
save(
    pd.DataFrame([{
        "matches":   matches,
        "runs":      runs,
        "balls":     balls,
        "wickets":   wickets,
        "avg_score": avg_score,
    }]),
    "summary_metrics",
)

# ── 4. ML ─────────────────────────────────────────────────────────────────────
print("\n[ML] Feature engineering...")
ml_df = sdf.withColumn(
    "label",
    F.when(F.col("runs_batter").isin(4, 6), 1).otherwise(0).cast(FloatType()),
)

cat_cols = ["batting_team", "bowling_team", "bowler", "batter"]
for c in cat_cols:
    ml_df = ml_df.fillna({c: "Unknown"})

feature_cols = ["over", "ball", "bat_pos", "valid_ball"] + cat_cols
ml_df = ml_df.select(feature_cols + ["label"]).dropna()

idx_cols  = [c + "_idx" for c in cat_cols]
indexers  = [
    StringIndexer(inputCol=c, outputCol=o, handleInvalid="keep")
    for c, o in zip(cat_cols, idx_cols)
]
assembler = VectorAssembler(
    inputCols=["over", "ball", "bat_pos", "valid_ball"] + idx_cols,
    outputCol="features",
)

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"Train: {train_df.count():,}  |  Test: {test_df.count():,}")

models = {
    "Logistic Regression":   LogisticRegression(maxIter=20, featuresCol="features", labelCol="label"),
    "Decision Tree":         DecisionTreeClassifier(maxDepth=6, featuresCol="features", labelCol="label", maxBins=500),
    "Random Forest":         RandomForestClassifier(numTrees=50, maxDepth=6, seed=42, featuresCol="features", labelCol="label", maxBins=500),
    "Gradient Boosted Tree": GBTClassifier(maxIter=20, maxDepth=5, seed=42, featuresCol="features", labelCol="label", maxBins=500),
}

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
results = {}

for name, clf in models.items():
    print(f"  Training {name}...")
    t0    = time.time()
    pipe  = Pipeline(stages=indexers + [assembler, clf])
    model = pipe.fit(train_df)
    preds = model.transform(test_df)
    auc   = round(evaluator.evaluate(preds), 4)
    elapsed = round(time.time() - t0, 1)
    results[name] = {"AUC": auc, "Time_s": elapsed}
    print(f"    → AUC: {auc}  ({elapsed}s)")

results_df = (
    pd.DataFrame(results).T
    .reset_index()
    .rename(columns={"index": "Model"})
    .sort_values("AUC", ascending=False)
    .reset_index(drop=True)
)
save(results_df, "ml_results")

spark.stop()
print("\n✅  PySpark pipeline complete. All CSVs saved to:", args.out)

# ── 5. Train a lightweight sklearn model for live prediction ──────────────────
# (scikit-learn runs in the Streamlit app without Spark)
print("\n[Sklearn] Training Random Forest for live prediction page...")
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.preprocessing import LabelEncoder

# Reload from saved CSV (Spark already stopped)
raw = pd.read_csv(args.data)

# Keep only needed columns
needed = ["batting_team", "bowling_team", "batter", "bowler",
          "over", "ball", "bat_pos", "valid_ball", "runs_batter"]
raw = raw[needed].dropna()

# Label encode categoricals
encoders = {}
for col in ["batting_team", "bowling_team", "batter", "bowler"]:
    le = LabelEncoder()
    raw[col + "_enc"] = le.fit_transform(raw[col].astype(str))
    encoders[col] = le

# Target: boundary (4 or 6)
raw["label"] = raw["runs_batter"].isin([4, 6]).astype(int)

feature_cols = ["over", "ball", "bat_pos", "valid_ball",
                "batting_team_enc", "bowling_team_enc",
                "batter_enc", "bowler_enc"]

X = raw[feature_cols].values
y = raw["label"].values

clf = SkRF(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
clf.fit(X, y)

# Save model + encoders
model_path = os.path.join(args.out, "boundary_model.pkl")
joblib.dump({"model": clf, "encoders": encoders, "features": feature_cols}, model_path)
print(f"  Saved → {model_path}")

# Save dropdown lists for the UI
for col in ["batting_team", "bowling_team", "batter", "bowler"]:
    unique_vals = sorted(raw[col].dropna().unique().tolist())
    pd.DataFrame({col: unique_vals}).to_csv(
        os.path.join(args.out, f"unique_{col}.csv"), index=False
    )
    print(f"  Saved → unique_{col}.csv  ({len(unique_vals)} values)")

print("\n✅  All done! Both PySpark CSVs and sklearn model saved to:", args.out)