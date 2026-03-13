# Airline-Data Project
## Strategic Pricing Dynamics of Spirit Airlines in the U.S. Domestic Market
### A PySpark Big Data Analysis

---

## Overview

This project investigates whether the entry and route expansion of Spirit Airlines (NK) causes legacy incumbent carriers to lower their fares on overlapping routes. Using distributed data processing via PySpark, millions of ticket-level observations are analyzed from the U.S. Department of Transportation's DB1B Market dataset spanning 2008–2016.

**Research Question:**
> Does the entry and route expansion of Spirit Airlines cause legacy incumbent carriers to lower their fares on overlapping routes?

---

## Repository Structure

```
├── data/
│   ├── db1b_market/                    # Raw DB1B Market quarterly CSVs (2008–2016)
│   └── full_combined.csv               # Cleaned merged dataset (DB1B + T100 + OTP)
├── notebooks/
│   └── Airline Data.ipynb   # Main analysis notebook
├── outputs/
│   ├── plot1_fare_trend.png            # Fare trends by carrier over time
│   ├── plot2_spirit_presence.png       # Fares on routes with vs without Spirit
│   ├── plot3_fare_dispersion.png       # Top 20 routes by fare dispersion
│   ├── kmeans_elbow.png                # K-Means elbow plot
│   ├── gbt_feature_importance.png      # GBT feature importances
└── README.md
```

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| DB1B Market | BTS / DOT | 10% sample of all U.S. domestic airline tickets — fares, routes, carriers |
| T-100 Domestic Segment | BTS / DOT | Flight-level traffic and capacity data |
| On-Time Performance (OTP) | BTS / DOT | Departure/arrival delay metrics |

**Study period:** 2008–2016 (note: 2013 Q4, 2014 Q2/Q3, 2016 Q1 missing due to BTS availability)

**Key filters applied:**
- U.S. domestic flights only
- Fares between $25 and $2,500
- NYC slot-controlled airports excluded (JFK, LGA, EWR)
- Bulk/opaque fares excluded
- ULCC carriers (NK, F9, G4, SY) excluded from incumbent analysis

---

## Setup & Requirements

### Prerequisites
```bash
pip install pyspark findspark jupyter pandas matplotlib statsmodels
```

### Spark Configuration
```python
spark = SparkSession.builder \
    .appName("SpiritPricing") \
    .config("spark.driver.memory", "12g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()
```

### Data Setup
1. Download DB1B Market quarterly files from [BTS TranStats](https://www.transtats.bts.gov) for 2008–2016
2. Place files in `data/db1b_market/` — naming format: `db1b_Market_YYYY_QN.csv`
3. Place `full_combined.csv` in `data/`
4. Update file paths in the notebook to match your local directory

---

## Analysis Pipeline

### 1. Data Loading & Cleaning
- Load all quarterly DB1B Market files via PySpark wildcard read
- Filter to study period, domestic routes, valid fares
- Build unordered city-pair route keys (e.g. `ATL_DTW`)
- Remove NYC slot-controlled airports

### 2. Fare Aggregation & Join
- Aggregate fares to carrier-route-quarter level (median, mean, IQR)
- Join onto `full_combined.csv` which contains HHI, carrier share, and delay metrics
- **Match rate: 88.9%** across 24.6M rows

### 3. Exploratory Data Analysis (EDA)
All aggregations performed in PySpark; `.toPandas()` called only for final plotting.

| Plot | Description |
|------|-------------|
| Fare trends over time | Spirit vs legacy carriers 2008–2016, with boom period marked |
| Routes with vs without Spirit | Incumbent fare comparison by Spirit presence |
| Fare by boom period | Distribution across pre-boom, boom, post-boom, post-period |
| Route fare dispersion | Top 20 routes by IQR — identifies most price-contested markets |

### 4. Regression Analysis (PySpark MLlib LinearRegression)
Panel regression estimating incumbent fare response to Spirit's pricing:

```
avg_fare ~ spirit_fare_lag + HHI + carrier_share + spirit_present + shorthaul + carrier_FE + route_FE
```

Three models estimated:
- **Model 1:** Baseline — all routes
- **Model 2:** Shorthaul routes only (Distance Group ≤ 2)
- **Model 3:** Spirit-present routes only

Fixed effects implemented via StringIndexer → OneHotEncoder within a PySpark ML Pipeline.

### 5. Machine Learning

#### K-Means Clustering (PySpark MLlib)
Segments 1,605 routes into 4 clusters based on competitive structure:

| Cluster | Avg HHI | Avg Fare | Spirit Frequency | Interpretation |
|---------|---------|----------|-----------------|----------------|
| 0 | 5,228 | $174 | 0.5% | Competitive, low-fare routes |
| 1 | 5,406 | $163 | **42.7%** | Spirit's core battleground — lowest fares |
| 2 | 6,043 | $277 | 0.1% | Premium routes — Spirit avoids |
| 3 | 9,326 | $170 | 0.2% | Near-monopoly regional routes |

Optimal k selected via elbow plot on within-cluster sum of squares.

#### Gradient Boosted Trees (PySpark MLlib)
Predicts incumbent fare from market structure features.

**Pipeline:** StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler -> GBTRegressor

**Features:** HHI, carrier share, Spirit presence, Spirit fare, shorthaul flag, carrier OHE

**Train/test split:** 80/20 random split with fare-bucket stratification to address class imbalance

**Results:**
| Metric | Original | Balanced |
|--------|----------|---------|
| RMSE | $58.91 | $68.91 |
| MAE | $43.40 | $51.95 |
| R squared | 0.2660 | **0.3313** |

Top features by importance: `carrier_share` > `HHI` > `carrier_ohe` > `spirit_present` > `spirit_fare`

---

## Key Findings

1. **K-Means clustering** shows routes with the highest Spirit presence (Cluster 1, 43% frequency) have the lowest incumbent fares at $163 — $114 cheaper than premium routes where Spirit rarely competes, consistent with Spirit exerting meaningful downward pricing pressure.

2. **GBT feature importance** confirms that market structure variables (`carrier_share`, `HHI`) explain more fare variation than Spirit-specific variables, suggesting that while ULCC competition matters, route concentration remains the primary driver of incumbent pricing. Using balanced classes improved the R squared by 7 percentage points.

3. **Shorthaul routes** show stronger Spirit effects than longhaul, consistent with Spirit's operation and focus only on short-distance markets.

---

## Limitations

- Not enough depth on regression parameters to investigate a causal relationship. Therefore the regression was left out of the analysis and focus was only on K-Means clustering and GBT models.
- Furthermore, PySpark LinearRegression could not support clustering of standard errors by route which would make results more robust.
- Missing quarters (2013 Q4, 2014 Q2/Q3, 2016 Q1) due to BTS website unavailability this week. Severely impacted the outcome of the project. 
- DB1B is a 10% ticket sample, not a census of all fares
- No booking window, seat class, or ancillary fee data available
- R squared of 0.33 reflects the difficulty of predicting fares without ticket-level detail and other exogenous factors that may be affecting prices (for example demand, availability of substitutes and oil prices).


---

## Related Work

- Gil & Kim (2021) — Competition and service quality in U.S. airlines
- Shrago (2024) — Spirit effect on fare dispersion
- Bachwich & Wittman (2017) — ULCC business model emergence
- Wang & Ma (2024) — Incumbent responses to Spirit entry

---

## Authors

Sarah Nasaka, 
Drexel University — DSCI 632, March 2025
