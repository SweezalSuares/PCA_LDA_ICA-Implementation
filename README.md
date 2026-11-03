# Dimensionality Reduction on Breast Cancer Data (PCA · LDA · ICA)

Streamlit app built from **Lab 4** (Sweezal Suares, 547). It explores the Breast Cancer
Wisconsin (Diagnostic) dataset from scikit-learn and compares three dimensionality-reduction methods.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `Sweezal_Suares_547_lab4.ipynb` | Lab notebook (cleaned so it runs top-to-bottom) |
| `requirements.txt` | Python dependencies |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
On Windows, use `python -m streamlit run app.py` if `streamlit` is not on PATH.

## App sections
- **Data Explorer** – preview, summary statistics, class balance, per-feature histograms
- **PCA** – explained-variance chart, components needed for 90%/95% variance, component scatter plots
- **LDA** – 1-D supervised projection with class histograms
- **ICA** – independent-component scatter plot and optional pairplot
- **Comparison** – PCA vs LDA vs ICA side by side

Sidebar controls set the number of PCA/ICA components and the ICA random seed.

## Notebook fixes vs. original
- Defined `df_cancer`, `y`, `data` (previously undefined variables)
- 90%/95% variance counts now use full PCA (previously only 4 components were fitted, so the answer was wrong)
- Comparison plots now create their `axes` with `plt.subplots`
