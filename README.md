# H&M Personalized Fashion Recommendations

An end-to-end recommendation system built on the [H&M Personalized Fashion
Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset — covering exploratory data analysis, RFM customer segmentation, and
a two-stage recommendation pipeline (candidate generation + learning-to-rank)
evaluated with **MAP@12**, memory-optimized to run on Google Colab's free tier.

---

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Phase A — EDA & Customer Segmentation](#phase-a--eda--customer-segmentation)
- [Phase B — Recommendation Modeling](#phase-b--recommendation-modeling)
- [Models Used & How They Work](#models-used--how-they-work)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

## Dataset

Three linked tables:

| Table | Size | Description |
|---|---|---|
| `transactions` | ~31.7M rows | Every purchase: customer, article, date, price, sales channel |
| `articles` | ~105K rows | Product metadata: category, color, garment group, department |
| `customers` | ~1.37M rows | Customer metadata: age, membership status, news subscription |

---

## Project Structure

```
├── notebook.ipynb          # Full pipeline: Phase A (EDA/RFM) + Phase B (modeling)
├── data/                    # (not committed) raw + intermediate parquet files
└── README.md
```

---

## Phase A — EDA & Customer Segmentation

**A.1 — Memory-efficient data loading.** Datasets are converted from Hugging
Face format to pandas exactly once, with the original Hugging Face objects
deleted immediately afterward to avoid holding duplicate copies in memory.

**A.2 — Dtype optimization.** `customer_id` and `article_id` are converted to
`category` dtype (they're long strings repeated millions of times — this is
the single biggest memory saving available), prices to `float32`, and channel
IDs to `int8`.

**A.3–A.4 — Data quality checks & EDA.** Missing values, duplicates, and
distribution checks across product categories and price.

**A.5 — RFM Segmentation.** Each customer is scored on:
- **Recency** — days since last purchase
- **Frequency** — number of purchases
- **Monetary** — total amount spent

Each dimension is split into quintiles (1-5) and combined into a segment
label (*Champions, Loyal Customers, At Risk, Lost Customers*, etc.) — a
standard marketing-analytics technique for understanding *who* your
customers are before trying to predict *what* they'll buy next.

---

## Phase B — Recommendation Modeling

The core insight driving this phase: **recommending products is a ranking
problem, not a classification problem.** Predicting "will this customer buy
this specific product: yes/no" independently for every product ignores the
fact that we only get to show 12 recommendations, and their *order* matters.
That's why every stage below is built around producing an ordered list and
evaluating it with **MAP@12** (Mean Average Precision at 12) — the same
metric the original Kaggle competition used.

**B.1 — Time-based train/validation split.** The last 6 days of transactions
are held out as validation. A random split isn't valid here — it would leak
future purchases into training. A recent 90-day rolling window (rather than
the full 2-year history) is used for modeling, which is far lighter on
memory and barely affects quality since recent behavior is most predictive
of near-term purchases anyway.

**B.2 — Candidate generation.** Scoring all 105K articles for every customer
isn't computationally feasible, so a shortlist (~30 items) is built per
customer from three complementary sources:
- items they've bought before (**repurchase**)
- current bestsellers within their favorite product category
- items suggested by the ALS collaborative-filtering model (see below)

**B.3 — Feature engineering.** Each (customer, candidate item) pair gets a
handful of signals: item popularity, average price, the customer's purchase
frequency and average spend, product category, and the ALS affinity score.

**B.4 — Ranking.** A LightGBM Ranker learns to order the ~30 candidates per
customer from most to least likely to be purchased.

---

## Models Used & How They Work

### 1. Popularity Baseline *(no training — a business rule)*
Recommends the 12 best-selling articles from the last 14 days to every
customer. This isn't a model, but it's an essential comparison point: if a
"smart" model can't beat this, the added complexity isn't earning its keep.

### 2. ALS — Alternating Least Squares *(collaborative filtering)*
Think of a giant grid: rows = customers, columns = articles, with a mark
wherever someone bought that item. ALS factorizes this grid into two smaller
sets of numbers — a "taste vector" for every customer and a matching
"style vector" for every article. When a customer's taste vector lines up
well with an article's style vector, ALS predicts they'd like it — even if
they've never bought anything similar before. It's the mathematical version
of "customers like you also bought this," applied across the whole customer
base simultaneously. Used here both to generate extra candidates and to
provide an `als_score` feature for the ranker below.

### 3. LightGBM Ranker (LambdaMART) *(learning-to-rank)*
Unlike a normal classifier, which scores each item independently, a
**ranker** is trained specifically to put the true purchase near the top of
an ordered list. It takes every feature we computed (ALS score, popularity,
price, category, customer spend habits) and learns how to weigh and combine
them so that, across thousands of training examples, the item the customer
actually bought tends to land near position 1 rather than position 30. This
is the same family of algorithm used by most real-world search and
recommendation ranking systems (it's the same core technique behind
learning-to-rank in web search).

---

## Results

Evaluated on a held-out validation set using **MAP@12**:

| Model | MAP@12 | Improvement over baseline |
|---|---|---|
| Popularity baseline | 0.0067 | — |
| ALS-only | 0.0070 | +4% |
| **LightGBM Ranker + ALS blend** | **0.0509** | **+660% (~7.5x)** |

### What this shows
- **ALS alone barely beats the popularity baseline.** Its raw top-N
  recommendations aren't dramatically better than simply showing bestsellers.
- **The real gain comes from blending signals, not from any single model.**
  Feeding the ALS score into LightGBM alongside popularity, price, and
  category features produces a ~7.5x improvement over either baseline —
  the ranker is learning to combine collaborative-filtering signal with
  content and behavioral signal in a way neither achieves alone.
- This is a common and important finding in real recommender systems:
  individual signals are often weak on their own; a learned combination of
  several *weak* signals frequently outperforms any one *strong-looking*
  signal in isolation.

*Note: these numbers come from an initial evaluation on a 2,000-customer
sample and a 90-day data window (chosen for Colab memory constraints). They
should be treated as directionally reliable rather than final — re-running
at a larger sample size (10,000-20,000+ customers) is recommended before
citing a specific number as a definitive result.*

---

## Key Engineering Decisions

Built and iterated specifically to run within Google Colab's free-tier
memory limit (~12GB RAM):

| Decision | Why |
|---|---|
| Convert IDs to `category` dtype immediately after loading | Largest single memory saving; avoids storing repeated long strings |
| Delete raw Hugging Face dataset objects right after conversion | Avoids holding 2-3 copies of a multi-GB table simultaneously |
| Model on a 90-day rolling window instead of full 2-year history | Comparable predictive quality, a fraction of the memory footprint |
| Vectorized ALS scoring (numpy dot product) instead of row-wise `.apply()` | `.apply()` is slow and memory-heavy in pandas; vectorized numpy ops process all rows at once |
| Batched evaluation (process customers in chunks, `del` + `gc.collect()` between batches) | Enables evaluating at 10,000+ customers without holding one giant candidate table in memory |
| `.map()` lookups instead of full table merges for single-column joins | Avoids duplicating every column of a large table just to fetch one value |

---

## How to Run

```bash
# In Google Colab
!pip install lightgbm implicit psutil --quiet
```
Run the notebook top to bottom. If memory gets tight partway through:
1. `Runtime > Restart runtime` after Phase A finishes — its outputs are saved
   to disk (`rfm.parquet`, `articles_slim.parquet`), so Phase B doesn't need
   Phase A's dataframes to still be in memory.
2. Reduce `WINDOW_DAYS` (default 90) or `SAMPLE_SIZE` / `EVAL_SIZE` if still
   constrained.
3. Use a High-RAM runtime (Colab Pro) if available.

---

## Future Improvements

- Re-run evaluation at full scale (all validation customers, full 2-year
  history) to confirm results hold beyond the memory-constrained sample.
- Add **recall@12** alongside MAP@12 to separately diagnose candidate
  generation quality vs. ranking quality.
- Tie RFM segments back into the recommender — e.g., report MAP@12 broken
  out by customer segment (Champions vs. At Risk) to connect the business
  analysis with the ML results.
- Add hybrid retrieval (BM25/text-based on product descriptions) as an
  additional candidate source.
- Promote the model artifact to a proper serving layer (FastAPI) with
  monitoring for prediction drift over time.
