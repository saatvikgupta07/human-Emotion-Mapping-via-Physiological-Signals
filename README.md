# Human Emotion Mapping via Physiological Signals

Classify a person's emotional state from their heartbeat.

This project trains a random forest to tell four states apart (baseline, stress, amusement and
meditation) using heart rate variability (HRV) features from the WESAD wearable-sensor dataset.
Everything is in one notebook, [`model.ipynb`](model.ipynb): exploring the data, cleaning it,
visualising it with PCA, training the model and ranking the features that matter most.

## Results

| Class | Precision | Recall | F1 | Test rows |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 1.00 | 1.00 | 1.00 | 7,644 |
| Stress | 1.00 | 1.00 | 1.00 | 4,389 |
| Amusement | 1.00 | 1.00 | 1.00 | 2,467 |
| Meditation | 1.00 | 1.00 | 1.00 | 5,198 |

All 19,698 test rows are classified correctly. Read
[How to interpret the 100%](#how-to-interpret-the-100) before relying on that number: it
measures something narrower than "detects emotion in a new person".

## Dataset

| | |
| :--- | :--- |
| Source | [`GeorgiaCh96/WESAD_raw_data`](https://huggingface.co/datasets/GeorgiaCh96/WESAD_raw_data) on the Hugging Face Hub |
| File | `WESAD_raw_data.csv` |
| Size | 196,978 rows × 66 columns |
| Contents | HRV features computed over time windows, plus `Time`, `subject id`, `condition` and `SSSQ` |
| Missing values | None |

WESAD (Wearable Stress and Affect Detection) records people wearing sensors while they go
through different conditions. Each row here is one time window, labelled with the condition the
person was in:

| Label | Condition | Rows |
| :--- | :--- | :--- |
| 0 | Baseline | 76,440 |
| 1 | Stress | 43,886 |
| 2 | Amusement | 24,676 |
| 3 | Meditation | 51,976 |

The classes are uneven, so the model is trained with `class_weight='balanced'`.

## What the notebook does

1. **Load** the CSV straight from the Hugging Face Hub.
2. **Explore** the label counts, missing values, box plots of the core signals (`HR`,
   `MEAN_RR`, `SDRR`, `RMSSD`), heart rate over time, heart rate per condition, and a
   correlation heatmap.
3. **Clean** the four core signals by capping values more than 3 standard deviations from the
   mean and applying a rolling median (window of 5). These cleaned columns feed the plots and
   the range check. The model's feature table is built before this step, so the model trains
   on the uncapped values.
4. **Prune features** by dropping any column correlated above 0.85 with an earlier one. This
   removes 38 of the 63 features and leaves 25.
5. **Scale** the remaining features with `StandardScaler`.
6. **Visualise with PCA** in 2D and 3D. PCA is used for the plots only; the model is trained
   on the 25 scaled features.
7. **Train** a random forest on a stratified 90/10 split (177,280 training rows, 19,698 test
   rows).
8. **Evaluate** with a classification report, a confusion matrix, feature importances and five
   sample predictions.

The model:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
```

## What the data shows

- **Heart rate separates stress from the rest.** Stress is the only condition with heart rates
  above roughly 110 bpm. Baseline, amusement and meditation overlap heavily between about 55
  and 90 bpm, so heart rate alone cannot tell those three apart.
- **Many features are redundant.** The dataset includes log, square-root, Box-Cox and
  Yeo-Johnson versions of the same measurements, which is why correlation pruning removes more
  than half of the columns.
- **The variance is spread over many directions.** The first principal component explains
  about 28% of the variance, the first five about 68%, and it takes 15 components to pass 95%.
- **The 2D projection shows a trend, not clean clusters.** Stress sits mostly on one side and
  meditation mostly on the other, with baseline and amusement overlapping both. The notebook
  reads the first component as arousal and the second as valence; that is an interpretation and
  is not tested.

## Most important features

Feature importances from the trained random forest:

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | `MEAN_RR` | 0.153 |
| 2 | `SDRR_RMSSD_REL_RR` | 0.095 |
| 3 | `SSSQ` | 0.093 |
| 4 | `LF_BOXCOX` | 0.085 |
| 5 | `MEDIAN_REL_RR` | 0.056 |
| 6 | `RMSSD_REL_RR` | 0.050 |
| 7 | `RMSSD` | 0.045 |
| 8 | `VLF_PCT` | 0.036 |
| 9 | `SDRR` | 0.034 |
| 10 | `VLF` | 0.034 |
| 11 | `SKEW` | 0.032 |
| 12 | `HR_LF` | 0.029 |
| 13 | `SDRR_RMSSD` | 0.029 |
| 14 | `LF_HF_LOG` | 0.029 |
| 15 | `LF_HF` | 0.028 |

A short guide to the names:

- `MEAN_RR` is the average time between heartbeats. It is the inverse of heart rate.
- `SDRR` and `RMSSD` measure how much that time varies, overall and from beat to beat.
- `VLF`, `LF` and `HF` are the power in the very-low, low and high frequency bands of the
  heartbeat signal. `LF_HF` is the ratio of low to high.
- `SSSQ` is a score from a stress questionnaire the participants filled in. It is not a sensor
  reading.

## How to interpret the 100%

The perfect score is real for this test set, but the test set is easy in three ways:

- **Neighbouring rows are near-copies.** The rows come from heavily overlapping time windows.
  The first three rows have a `MEAN_RR` of 654.01, 654.06 and 654.02, for example. A random
  split puts some of those rows in training and their neighbours in the test set, so the model
  has effectively seen each test row already.
- **The same people are in both sets.** The split is by row, not by person, so the model is
  never asked about someone it has not seen.
- **`SSSQ` is a questionnaire answer.** It is the third most important feature, and it would
  not be available from a sensor alone.

So the result shows that the model can recognise windows from recordings it was trained on. It
does not show how well it would detect emotion in a new person. Testing that would mean holding
out whole subjects (the `subject id` column makes this possible) and dropping `SSSQ`.

## Running it

Install the dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn huggingface_hub jupyter
```

Then open the notebook and run the cells in order:

```bash
jupyter notebook model.ipynb
```

Things to know before you run it:

- The dataset is downloaded from the Hugging Face Hub on first run, so you need an internet
  connection.
- The random forest trains 300 trees on about 177,000 rows using every CPU core.
- The repository has no pinned requirements file. The notebook was last run on Python 3.11.
