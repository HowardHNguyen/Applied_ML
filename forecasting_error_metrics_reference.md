# Forecasting Error Metrics: RMSE, MAE, MAPE, and sMAPE

*A reference note on the four most common error metrics used to evaluate point forecasts — what each one measures, when to use which, and how to read them honestly.*

---

## Why error metrics matter

Every forecasting study eventually comes down to one question: **how wrong were the predictions, on average?** That question has multiple defensible answers, and the choice of metric changes which model "wins." Reporting only one metric is incomplete; reporting all four is good practice, and understanding why they sometimes disagree is essential for honest reporting.

A simple way to think about the four metrics:

| Metric | Units | What it emphasizes |
|---|---|---|
| **RMSE** | Same as the data | Large errors (squared, then rooted) |
| **MAE** | Same as the data | Average size of error, all equally weighted |
| **MAPE** | Percent (%) | Relative error vs. true value |
| **sMAPE** | Percent (%) | Relative error, symmetric, bounded |

---

## 1. RMSE — Root Mean Squared Error

### Definition

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and $n$ is the number of forecasted points.

### What it measures

The square root of the average squared error. Because errors are **squared before averaging**, large errors contribute disproportionately. An error of 10 contributes 100 to the sum; an error of 100 contributes 10,000.

### Properties

- **Units:** Same as the target variable. If you are forecasting ILI cases, RMSE is in cases. If forecasting dollars, RMSE is in dollars.
- **Range:** $[0, \infty)$. Zero means perfect prediction.
- **Sensitivity to outliers:** **High.** A single big miss can dominate the score.
- **Differentiability:** Smooth and differentiable, which is why squared error is the basis of most ML loss functions (MSE = RMSE²).

### When to use it

- When **large errors are disproportionately costly** — e.g., underforecasting flu cases by 10,000 is more than 10× worse than underforecasting by 1,000 because hospital systems get overwhelmed nonlinearly
- When you want a metric that aligns with **statistical training objectives** (most regression models minimize squared error)
- When the data has **no zero or near-zero values** that would distort percentage-based metrics

### When NOT to use it

- When comparing forecasts across series of very different scales (RMSE for a series with values around 100,000 will look much larger than RMSE for a series around 100 — but neither model is necessarily worse)
- When stakeholders need a percentage-style interpretation

### Example

If your forecast errors are $[-10, +5, +5, +100]$:
- Squared: $[100, 25, 25, 10000]$
- Mean: $2537.5$
- RMSE: $\sqrt{2537.5} \approx 50.4$

Note how the single error of 100 dominates the score, even though three of four predictions were nearly correct.

---

## 2. MAE — Mean Absolute Error

### Definition

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### What it measures

The average absolute error. Each miss counts equally regardless of magnitude — no squaring, no penalization of big errors.

### Properties

- **Units:** Same as the target variable.
- **Range:** $[0, \infty)$. Zero means perfect prediction.
- **Sensitivity to outliers:** **Low.** A single big miss contributes its magnitude, not its square.
- **Interpretation:** "On average, my forecast was off by X units."

### When to use it

- When you want an **intuitive, easy-to-explain** metric ("we were off by 2,000 cases on average")
- When **all errors are equally important** regardless of size — for example, in inventory planning, being off by 50 units in either direction has roughly linear consequences
- When the data has **outliers you do not want to dominate** the evaluation

### When NOT to use it

- When large errors really are disproportionately bad (use RMSE)
- For ML model training — MAE's gradient is constant, which can make optimization slower than MSE

### Example

Same errors $[-10, +5, +5, +100]$:
- Absolute: $[10, 5, 5, 100]$
- Mean: $30$
- MAE: $30$

Notice that MAE (30) is much smaller than RMSE (50.4) for the same errors. **A large gap between MAE and RMSE indicates outlier-driven errors.** If MAE ≈ RMSE, errors are evenly sized; if RMSE >> MAE, a few large misses are dominating.

### Rule of thumb

For any error distribution, $\text{MAE} \leq \text{RMSE}$, always. They are equal only if every error has the same magnitude.

---

## 3. MAPE — Mean Absolute Percentage Error

### Definition

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

### What it measures

The average percentage error relative to the actual value. Expressed as a percent, so it is scale-free and comparable across series.

### Properties

- **Units:** Percent (%). Independent of the scale of the data.
- **Range:** $[0, \infty)$. Yes, MAPE can exceed 100% — e.g., if you predict 200 when the truth is 50, the percent error is 300%.
- **Sensitivity to outliers:** Moderate, but **highly sensitive to small actual values** (the denominator).
- **Interpretation:** "On average, my forecast was off by X% of the true value."

### When to use it

- When comparing forecasts **across series of different scales** (revenue vs. units vs. visits)
- When **relative error matters more than absolute error** — being off by 10% of $1M is materially worse than being off by 10% of $100, even though the absolute error is larger in the first case
- When communicating to **non-technical stakeholders** who think in percentages

### When NOT to use it

- **When actual values can be zero or near zero** — MAPE is undefined when $y_i = 0$ and explodes when $y_i$ is small. A flu surveillance series that dips to 5 cases in summer will produce huge MAPE values for tiny absolute errors.
- **When you want symmetric treatment of over- and under-forecasting** — MAPE is asymmetric (see next section).

### The asymmetry problem

MAPE penalizes over-forecasting more than under-forecasting:

- True = 100, Forecast = 200: MAPE = $|100-200|/100 = 100\%$
- True = 100, Forecast = 0: MAPE = $|100-0|/100 = 100\%$
- True = 100, Forecast = 50: MAPE = $|100-50|/100 = 50\%$
- True = 50, Forecast = 100: MAPE = $|50-100|/50 = 100\%$

The last two examples show the asymmetry: predicting 100 when truth is 50 (over by 50) is penalized twice as harshly as predicting 50 when truth is 100 (under by 50). This makes MAPE biased toward forecasts that lean low.

### Example

Actuals $[100, 50, 200, 1000]$, predictions $[110, 55, 180, 900]$:
- Percent errors: $[10\%, 10\%, 10\%, 10\%]$
- MAPE: $10\%$

Now if one actual is small, like $[100, 50, 200, 5]$ with the same predictions $[110, 55, 180, 9]$:
- Percent errors: $[10\%, 10\%, 10\%, 80\%]$
- MAPE: $27.5\%$

The single small-actual point dragged the average up dramatically, even though the absolute error there (4) is the smallest of any prediction.

---

## 4. sMAPE — Symmetric Mean Absolute Percentage Error

### Definition

$$\text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

### What it measures

A symmetric version of MAPE that divides by the **average of the actual and predicted values**, rather than just the actual. Designed to handle MAPE's asymmetry and small-denominator problems.

### Properties

- **Units:** Percent (%).
- **Range:** $[0\%, 200\%]$. Bounded above, which is a major practical advantage.
- **Sensitivity to outliers:** Lower than MAPE; the denominator cannot go to zero unless both actual AND predicted are zero.
- **Symmetry:** Over- and under-forecasting by the same absolute amount produce similar (but not exactly equal — see below) sMAPE values.
- **Interpretation:** Roughly "average percent error, on a 0-200% scale."

### When to use it

- When the data has **zero or near-zero values** that would distort MAPE
- When you want a **bounded percent error** that cannot blow up
- When you need **comparability across series** but are uncomfortable with MAPE's asymmetry
- This is the **default metric in the M4 and M5 forecasting competitions** for exactly these reasons

### When NOT to use it

- When stakeholders are already familiar with MAPE and the data has no small-value issues — sMAPE is harder to explain ("symmetric percent error averaged against the mean of true and predicted...")
- When **strict symmetry matters** — sMAPE is *more* symmetric than MAPE but not perfectly symmetric (some authors argue this defeats the purpose)

### The not-quite-symmetric problem

Even sMAPE is not perfectly symmetric. Consider:

- True = 100, Forecast = 110: sMAPE = $|10| / 105 = 9.52\%$
- True = 110, Forecast = 100: sMAPE = $|10| / 105 = 9.52\%$ ✅ symmetric

But:

- True = 100, Forecast = 0: sMAPE = $|100| / 50 = 200\%$
- True = 0, Forecast = 100: sMAPE = $|100| / 50 = 200\%$ ✅ symmetric

- True = 100, Forecast = 200: sMAPE = $|100| / 150 = 66.7\%$
- True = 200, Forecast = 100: sMAPE = $|100| / 150 = 66.7\%$ ✅ symmetric

So sMAPE IS symmetric in pairwise comparison — better than MAPE. The "not perfectly symmetric" criticism applies to edge cases involving zeros, where the formula's behavior is debatable.

### Example

Actuals $[100, 50, 200, 5]$, predictions $[110, 55, 180, 9]$:
- Denominators: $[105, 52.5, 190, 7]$
- Percent errors: $[9.5\%, 9.5\%, 10.5\%, 57.1\%]$
- sMAPE: $21.7\%$

Compare to MAPE on the same data: 27.5%. The small-value point still drives the metric up, but less violently than under MAPE.

---

## How to read the four metrics together

The single most useful diagnostic is to **report all four metrics and look for disagreements between them.** Each disagreement tells you something:

### RMSE much greater than MAE

**Diagnosis:** A few large errors are dominating. Your model probably misses badly on specific points (peaks, troughs, outliers) while handling the average case well.

**Example:** A flu forecaster that handles low-activity weeks fine but blows the January peak.

**Action:** Investigate which points have the largest errors. Consider whether those points matter operationally more than average performance.

### RMSE close to MAE

**Diagnosis:** Errors are evenly sized. The model is consistently slightly off across most predictions, with no single catastrophic miss.

**Action:** Either you have a well-calibrated but imperfect model, or your test data lacks the volatility to surface outlier behavior.

### MAPE much greater than sMAPE

**Diagnosis:** Some actual values in the test set are small, and MAPE is being inflated by those small denominators. The model might be fine; the metric is unfair.

**Example:** Forecasting flu surveillance counts that dip to single digits during summer. MAPE will look terrible for those weeks; sMAPE will be more reasonable.

**Action:** Trust sMAPE over MAPE in this case, or filter the test set to weeks where actuals exceed some threshold.

### sMAPE much higher than expected given RMSE/MAE

**Diagnosis:** Forecasts are off in *relative* terms even when absolute terms look fine. This often happens when the model predicts everything near the mean — absolute errors look acceptable but percentage errors on low-volume periods are catastrophic.

**Example:** An LSTM that learns to predict the long-run mean. RMSE and MAE look mediocre; sMAPE reveals the model is essentially useless during low-activity weeks.

---

## Worked example from the California ILI study

The following are real results from a 104-week test set, comparing forecasts for California ILI:

| Model | RMSE | MAE | MAPE | sMAPE |
|---|---|---|---|---|
| Naive Seasonal | 3,139 | 2,052 | 18.9% | 17.7% |
| SARIMA | 4,123 | 3,217 | 35.6% | 32.3% |
| XGBoost | 4,366 | 3,338 | 36.3% | 33.4% |
| ARIMA | 4,733 | 3,594 | 38.2% | 35.9% |
| LSTM | 5,063 | 4,419 | 57.3% | 43.3% |
| Prophet | 9,016 | 6,704 | 74.9% | 48.2% |

### What this table tells us

- **Ranking is identical across all four metrics.** This is strong evidence that the conclusion (Naive Seasonal wins) is robust, not a metric artifact.
- **RMSE > MAE by ~50% for most models.** Errors are uneven; the test set includes some big misses (likely the peaks).
- **MAPE > sMAPE for every model.** The test set contains some low-value weeks where MAPE is being inflated. sMAPE is the more honest percent-error metric here.
- **LSTM and Prophet have huge MAPE/sMAPE gaps.** This is a sign that they predict near the mean — their absolute errors are not catastrophic, but their relative errors during low-activity periods are.

### What to report

For this study, the right presentation is:

> *"Naive Seasonal achieved RMSE 3,139, MAE 2,052, and sMAPE 17.7% on a 104-week test set, beating the next-best model (SARIMA) by 24% on RMSE. The ranking is consistent across all four standard error metrics."*

That sentence is honest, complete, and pre-empts most reviewer questions.

---

## Recommended practice

### 1. Always report at least RMSE + MAE + sMAPE

These three together cover absolute-large-error sensitivity (RMSE), absolute-average sensitivity (MAE), and relative scale-free interpretability (sMAPE). MAPE is fine to add as a fourth if your data has no zero/near-zero values.

### 2. Pair the headline metric with at least one cross-check

If you lead with RMSE, also report MAE so readers can see whether outliers dominate. If you lead with MAPE, also report sMAPE so readers can see whether asymmetry distorts the picture.

### 3. Match the metric to the cost structure

- Inventory planning, where stockouts and overstocks have linear costs → **MAE**
- Hospital surge planning, where being off by 100 cases is far worse than being off by 10 → **RMSE**
- Cross-business reporting where different products have different scales → **sMAPE**
- Single product, well-understood scale, no zero values → **MAPE** (with caveats noted)

### 4. Report the test window and split

Any metric value is only meaningful in the context of the test window. "RMSE 3,139" means nothing on its own; "RMSE 3,139 on a 104-week held-out test set covering 2024-2026" means everything.

### 5. Be honest about in-sample vs out-of-sample

If you tuned anything on the test set (hyperparameters, ensemble weights, feature selection), report this clearly. Reporting test-set-tuned RMSE as "out-of-sample" is one of the most common forms of accidental dishonesty in forecasting.

---

## Quick decision flow

```
Are there zero or near-zero actual values?
├── YES → Use sMAPE (not MAPE)
└── NO
    │
    Do large errors matter disproportionately?
    ├── YES → Lead with RMSE
    └── NO
        │
        Do you need to compare across different scales?
        ├── YES → Lead with sMAPE or MAPE
        └── NO → Lead with MAE (easiest to explain)
```

---

## Summary table

| Property | RMSE | MAE | MAPE | sMAPE |
|---|---|---|---|---|
| Formula | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | $\frac{1}{n}\sum|y - \hat{y}|$ | $\frac{100}{n}\sum\frac{|y - \hat{y}|}{|y|}$ | $\frac{100}{n}\sum\frac{|y - \hat{y}|}{(|y|+|\hat{y}|)/2}$ |
| Units | Data units | Data units | Percent | Percent |
| Range | $[0, \infty)$ | $[0, \infty)$ | $[0, \infty)$ | $[0\%, 200\%]$ |
| Outlier sensitivity | High | Low | Moderate | Moderate |
| Handles zeros? | Yes | Yes | **No** | Yes |
| Symmetric? | Yes | Yes | **No** | Mostly |
| Easy to explain? | Moderate | **Easy** | **Easy** | Hard |
| Comparable across scales? | No | No | Yes | Yes |
| Used in ML training? | **Yes (MSE)** | Sometimes | Rarely | Rarely |
| Used in competitions? | Common | Common | Discouraged | **M4/M5 default** |

---

## Implementation reference

### Pure Python

```python
import numpy as np

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)
```

### Using scikit-learn

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
mae_value = mean_absolute_error(y_true, y_pred)
# MAPE: scikit-learn 0.24+ has mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error
mape_value = mean_absolute_percentage_error(y_true, y_pred) * 100
# sMAPE is not in scikit-learn — use the implementation above
```

### Using `statsmodels` and time-series libraries

Most time-series libraries (Nixtla's `utilsforecast`, the `forecast` package in R, `darts` in Python) provide all four metrics built-in. For one-off comparisons across multiple models, these libraries handle the bookkeeping automatically.

---

## Common mistakes to avoid

### 1. Reporting only one metric

Different metrics can disagree. Reporting only the one your model wins on is selective reporting. Always report at least two, ideally three or four.

### 2. Using MAPE on data with zeros

MAPE is undefined at zero and unstable near zero. Switch to sMAPE or filter out zero-actual points (and note that you did).

### 3. Comparing RMSE across different time series

RMSE of 1000 on a series that ranges 0-100,000 is excellent; RMSE of 1000 on a series that ranges 0-1,000 is a disaster. Always interpret RMSE relative to the data's typical magnitude. Use sMAPE or MAPE for cross-series comparison.

### 4. Cherry-picking the test window

A model that looks great on one test window may look terrible on another. Use walk-forward cross-validation to get a distribution of metric values, not just one point estimate.

### 5. Tuning on the test set

If you select hyperparameters, ensemble weights, or feature sets based on test-set performance, your reported metric is optimistically biased. Either hold out a separate validation set, or be explicit that you tuned on the test set and the reported number is in-sample for that tuning.

---

## References

- **Hyndman, R. J., & Koehler, A. B.** (2006). "Another look at measures of forecast accuracy." *International Journal of Forecasting,* 22(4), 679-688. The standard reference on forecast metric comparison.
- **Hyndman, R. J., & Athanasopoulos, G.** *Forecasting: Principles and Practice* (3rd ed.). OTexts. Chapter 5 covers evaluation metrics in depth. Free online: <https://otexts.com/fpp3/accuracy.html>
- **Makridakis, S.** (1993). "Accuracy measures: theoretical and practical concerns." *International Journal of Forecasting,* 9(4), 527-529. The original critique of MAPE that motivated sMAPE.
- **Makridakis, S., Spiliotis, E., & Assimakopoulos, V.** (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting.* Provides extensive empirical comparison of metrics across many forecasting tasks.

---

*This reference note accompanies the California ILI multi-model forecasting study, where ranking models by RMSE, MAE, MAPE, and sMAPE all produced identical orderings — providing strong evidence that the conclusions are metric-robust.*
