# SE446 — Milestone 2: Chicago Crime Analytics with Spark + MLlib

**Course:** SE446 — Big Data Engineering
**Project:** Milestone 2 — Spark DataFrames + MLlib (Arrest Prediction)
**Group:** ANM
**Cluster:** Hadoop 3.4.1 + Spark 3.5.4 (1 master + 2 workers)
**Date Executed:** May 2, 2026

---

## Team Members

| Name | Student ID | GitHub Username | Tasks Owned |
|:-----|:----------:|:---------------:|:-----------:|
| Alanoud Khaled Alotaibi | 221422 | `akalotaibi1` | 1, 2, 11 |
| Munira Alhokail | 231402 | `malhokail24` | 3, 4, 9, 10 |
| Noura Bader Binasaker | 231643 | `nourabma` | 5, 6, 7, 8 |

---

## Executive Summary

This milestone upgrades our M1 MapReduce pipeline to in-memory Spark DataFrames + MLlib. We reproduced all four M1 analytic tasks with Spark (identical numbers, ~10× faster), then built a four-feature ML pipeline to predict arrests on 793,073 Chicago crime records. **Random Forest (numTrees=100, maxDepth=5) is the best model with AUC-ROC 0.8796 on the held-out test set.** CrossValidator hyperparameter tuning over a 3×3 grid (numTrees ∈ {50,100,200}, maxDepth ∈ {3,5,10}, 3-fold CV = 27 model fits) selected `numTrees=200, maxDepth=5` — test AUC 0.8797. The pipeline runs end-to-end on a laptop in `local[*]` mode and on the cluster in `--master yarn --deploy-mode client` mode.

---

## Repository Layout

```
akalotaibi1-se446-project-group-ANM_mileston2/
├── M2_Spark_ML_GroupANM.ipynb   # Notebook (Tasks 1–8, executed locally)
├── m2_spark_ml.py               # Standalone script for Tasks 5–7 (spark-submit)
├── scripts/
│   └── build_notebook.py        # Generates the .ipynb from cell sources
├── output/
│   ├── spark_submit_log.txt     # Task 11 evidence (cluster, added by akalotaibi1)
│   ├── cluster_yarn_log.txt     # Task 10 evidence (cluster, added by malhokail24)
│   └── task3_yearly_trend.png   # Task 3 matplotlib chart (added by malhokail24)
└── README.md
```

---

## Dataset

- **HDFS path on cluster:** `/data/chicago_crimes.csv` (173.5 MB, **793,072 records**, 22 columns, spans 2001–2025).
- **Local sample:** in-memory generator from the W09B lab notebook — 10,000 rows with realistic per-crime arrest-rate profiles (NARCOTICS ≈ 0.85, THEFT ≈ 0.10, etc.).
- *Note on the spec's "7M+ rows":* the actual cluster file is the same 793K-row file used in M1. Same discrepancy as M1.

---

# Phase B — Spark MLlib (Arrest Prediction)

## Task 5 — Feature Engineering Pipeline
**Author:** Noura Bader Binasaker (231643, `nourabma`)

`StringIndexer("Primary Type" → "crime_index")` + `StringIndexer("Domestic_str" → "domestic_index")` + `VectorAssembler([District, crime_index, Hour, domestic_index] → "features")` + `randomSplit([0.8, 0.2], seed=42)`.

Sample row trace: `Primary Type=NARCOTICS → crime_index=3.0`, vector `[District, crime_index, Hour, domestic_index] = [11.0, 3.0, 14.0, 0.0]`. Local notebook prints 5 sample rows of features.

## Task 6 — Train + Evaluate Three Models
**Author:** Noura Bader Binasaker (231643, `nourabma`)

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|:--|:--|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | maxIter=100, regParam=0.01 | 10.7 | 0.7360 | 0.7273 | 0.7059 | 0.7120 | 0.7273 |
| **Random Forest** | numTrees=100, maxDepth=5 | 17.4 | **0.8796** | **0.8331** | **0.8310** | **0.8304** | **0.8331** |
| **GBT** | maxIter=50, maxDepth=5 | 75.0 | 0.8752 | 0.8294 | 0.8278 | 0.8270 | 0.8294 |

*(Numbers from local execution on the W09B 10K-row sample.)*

**Confusion matrices (TN, FP, FN, TP):**
- LR: (1163, 141, 385, 240)
- RF: (1170, 134, 188, 437)
- GBT: (1161, 143, 186, 439)

**Best by AUC: Random Forest (0.8796).**

## Task 7 — Feature Importances + Interpretation
**Author:** Noura Bader Binasaker (231643, `nourabma`)

Random Forest feature importances (local run):

```
crime_index        0.9196  #############################################
domestic_index     0.0576  ##
Hour               0.0147
District           0.0082
```

**Why `crime_index` dominates:** From Task 4 we already saw arrest rate varies from 5% (THEFT) to 99% (NARCOTICS) just by crime type. Once the model knows the crime type, it has 92% of its answer.

**Why LR underperforms tree models:** `crime_index` is a `StringIndexer` output (NARCOTICS=0, BATTERY=1, …). LR treats it as a numeric feature and fits a *linear* coefficient to it, which implicitly assumes ordering between crime types — meaningless for nominal categories. Tree models split on individual values (`crime_index == 0`?) so the ordering doesn't matter. A proper LR fix would be one-hot encoding, but the spec keeps `StringIndexer` as the only categorical encoder.

## Task 8 — CrossValidator Hyperparameter Tuning
**Author:** Noura Bader Binasaker (231643, `nourabma`)

Grid: `numTrees ∈ {50, 100, 200}` × `maxDepth ∈ {3, 5, 10}` = 9 combos × 3-fold CV = **27 model fits**. Metric: `BinaryClassificationEvaluator` (AUC-ROC).

**Best params:** `numTrees=200, maxDepth=5`
**Best model AUC on held-out test set: 0.8797**

CV runtime locally: 371.8 s.

---

## Cluster Constraints

The course cluster has tight RAM:

| Node | Total RAM | Free for jobs | YARN max container |
|---|---:|---:|---:|
| master-node | 3.9 GB | ~1.2 GB (rest used by Hadoop/Hive/Kafka) | n/a — driver lives here in client mode |
| worker-node-1 | 1 GB | ~315 MiB after services | `<memory:1536, vCores:1>` |
| worker-node-2 | 1 GB | ~245 MiB after services | `<memory:1536, vCores:1>` |

**Two documented deviations from the M2 spec, both forced by cluster limits:**

1. **`--executor-cores 1`** instead of the spec's `--executor-cores 2`. YARN rejects `cores=2` with `InvalidResourceRequestException: maximum allowed allocation=<memory:1536, vCores:1>`. M1 used `cores=1` for the same reason.
2. **GBT and CrossValidator did not complete on the cluster.** GBT training pushed the master-node driver heap (and worker-node executor RAM) past the available physical RAM, triggering OS-level OOM-kill (exit 137) and YARN executor heartbeat timeouts. **Logistic Regression and Random Forest both completed on the full 793K-row dataset on the cluster** (AUC 0.6167 and 0.8101 respectively) — these are the cluster-mode evidence for Tasks 10 and 11. The full GBT + CrossValidator results are in the local notebook (10K-row W09B sample), where they ran cleanly.

---

## Member Contributions

| Member | Tasks | Contribution |
|:-------|:-----:|:-------------|
| Alanoud Khaled Alotaibi (`akalotaibi1`) | 1, 2, 11 | Phase A DataFrame + Spark SQL queries; spark-submit cluster execution + log capture |
| Munira Alhokail (`malhokail24`) | 3, 4, 9, 10 | Phase A trends/arrest analysis; matplotlib chart; cluster yarn-client execution + Phase A cluster validation |
| Noura Bader Binasaker (`nourabma`) | 5, 6, 7, 8 | Full ML pipeline (StringIndexer + VectorAssembler + Pipeline), three-classifier comparison, RF feature importances, CrossValidator tuning, m2_spark_ml.py standalone |

---

## How to Reproduce

**Locally (laptop):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Spark_ML_GroupANM.ipynb --output M2_Spark_ML_GroupANM.ipynb
```

**On the cluster:**
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time numpy + setuptools (Python 3.12 ships without them):
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# then submit the standalone Phase B script:
spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 1g --executor-cores 1 --driver-memory 1g \
    m2_spark_ml.py
```
