# Dimensionality Reduction on Breast Cancer Data (PCA · LDA · ICA · t-SNE)

Streamlit app built from **Lab 4** (Sweezal Suares, 547). It explores the Breast Cancer
Wisconsin (Diagnostic) dataset from scikit-learn and compares four dimensionality-reduction methods.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `Sweezal_Suares_547_lab4(2).ipynb` | Current Lab 4 notebook |
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
- **t-SNE** – 2D embedding with adjustable perplexity
- **Comparison** – PCA vs LDA vs ICA vs t-SNE side by side

Sidebar controls set the number of PCA/ICA components, t-SNE perplexity, and random seed.

The PCA component counts for 90% and 95% variance are calculated from a full PCA fit,
so those counts remain valid even though the displayed projection uses a smaller number
of components.
