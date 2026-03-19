"""
ipl_pipeline.py  —  IPL PySpark Data Processing
=================================================
Generates all CSV files needed by the Streamlit dashboard.

Usage:
    python ipl_pipeline.py --data IPL.csv --out data

Requirements:
    pip install pyspark==3.5.1 pandas
"""

import os, argparse, warnings
import pandas as pd
warnings.filterwarnings("ignore")

# ── Java 17/21 compatibility fix ──────────────────────────────────────────────
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--conf \"spark.driver.extraJavaOptions="
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED\" "
    "pyspark-shell"
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="IPL.csv")
parser.add_argument("--out",  default="data")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

def save(df: pd.DataFrame, name: str):
    path = os.path.join(args.out, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  ✅  {name}.csv  ({len(df)} rows)")

# ── 1. Spark Session ───────────────────────────────────────────────────────────
print("Starting Spark...")
spark = (
    SparkSession.builder
    .appName("IPL_EDA")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.extraJavaOptions",
            "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
            "--add-opens=java.base/java.lang=ALL-UNNAMED")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print(f"Spark version: {spark.version}")

# ── 2. Load & cast ─────────────────────────────────────────────────────────────
print(f"\nLoading: {args.data}")
sdf = spark.read.csv(args.data, header=True, inferSchema=True)

int_cols = ["runs_batter","runs_extras","runs_total","balls_faced",
            "valid_ball","year","over","ball","runs_bowler"]
for c in int_cols:
    if c in sdf.columns:
        sdf = sdf.withColumn(c, F.col(c).cast(IntegerType()))

print(f"Total rows: {sdf.count():,}")
sdf.cache()

# ── 3. EDA aggregations ────────────────────────────────────────────────────────
print("\n[EDA] Running aggregations...")

# Season runs
save(
    sdf.groupBy("year")
       .agg(F.sum("runs_batter").alias("total_runs"))
       .orderBy("year").toPandas(),
    "season_runs"
)

# Wicket types
save(
    sdf.filter(F.col("wicket_kind").isNotNull())
       .groupBy("wicket_kind").count().toPandas(),
    "wicket_kinds"
)

# Match results
save(
    sdf.dropDuplicates(["match_id"])
       .groupBy("win_outcome").count().toPandas(),
    "match_results"
)

# Toss decisions
save(
    sdf.dropDuplicates(["match_id"])
       .groupBy("toss_decision").count().toPandas(),
    "toss_decisions"
)

# Top batters
save(
    sdf.groupBy("batter")
       .agg(F.sum("runs_batter").alias("runs"),
            F.sum("valid_ball").alias("balls"))
       .withColumn("SR", F.round(F.col("runs") / F.col("balls") * 100, 1))
       .orderBy(F.desc("runs")).limit(50).toPandas(),
    "top_batters"
)

# Runs by over
save(
    sdf.groupBy("over")
       .agg(F.sum("runs_batter").alias("runs_batter"))
       .orderBy("over").toPandas(),
    "runs_by_over"
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
       .orderBy(F.desc("wickets")).limit(50).toPandas(),
    "top_bowlers"
)

# Wickets by over
save(
    sdf.filter(F.col("wicket_kind").isNotNull())
       .groupBy("over")
       .agg(F.count("wicket_kind").alias("wicket_kind"))
       .orderBy("over").toPandas(),
    "wickets_by_over"
)

# Team runs
save(
    sdf.groupBy("batting_team")
       .agg(F.sum("runs_batter").alias("runs"),
            F.countDistinct("match_id").alias("matches"))
       .withColumn("RPM", F.round(F.col("runs") / F.col("matches"), 1))
       .orderBy(F.desc("runs")).toPandas(),
    "team_runs"
)

# Team trends by year
save(
    sdf.groupBy("year", "batting_team")
       .agg(F.sum("runs_batter").alias("runs_batter"))
       .orderBy("year").toPandas(),
    "team_trends"
)

# Season stats
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
       .orderBy("year").toPandas(),
    "season_stats"
)

# Sixes and fours per season
save(
    sdf.groupBy("year")
       .agg(
           F.sum(F.when(F.col("runs_batter") == 6, 1).otherwise(0)).alias("sixes"),
           F.sum(F.when(F.col("runs_batter") == 4, 1).otherwise(0)).alias("fours"),
       )
       .orderBy("year").toPandas(),
    "sixes_fours"
)

# Summary metrics
matches   = sdf.select("match_id").distinct().count()
runs      = int(sdf.agg(F.sum("runs_batter")).collect()[0][0] or 0)
balls     = int(sdf.agg(F.sum("valid_ball")).collect()[0][0] or 0)
wickets   = int(sdf.filter(F.col("wicket_kind").isNotNull()).count())
avg_score = round(
    sdf.groupBy("match_id").agg(F.sum("runs_total").alias("s"))
       .agg(F.avg("s")).collect()[0][0] or 0, 1)
save(
    pd.DataFrame([{"matches": matches, "runs": runs, "balls": balls,
                   "wickets": wickets, "avg_score": avg_score}]),
    "summary_metrics"
)

spark.stop()
print(f"\n✅  All done! {len(os.listdir(args.out))} files saved to: {args.out}/")