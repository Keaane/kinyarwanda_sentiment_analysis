# Kinyarwanda tweet sentiment classification

Three-way sentiment classification (positive / neutral / negative) of Kinyarwanda tweets, using the Kinyarwanda
subset of AfriSenti (Muhammad et al., 2023). Baselines (TF-IDF + logistic regression), Word2Vec + BiLSTM/BiGRU and
fine-tuned AfroXLMR (Alabi et al., 2022), with error analysis and a Streamlit demo.

## Links
- Live demo: **TODO: Streamlit Community Cloud URL**
- Model: https://huggingface.co/Gkea/kinyarwanda-sentiment-afroxlmr-large
- Demo video: **TODO**
- Report: **TODO**

## Data
`shmuhammad/AfriSenti-twitter-sentiment`, subset `kin`, official split: 3,302 train / 827 validation / 1,026 test
(one tweet fewer in each split than the dataset card lists, see notebook 01). Classes: neutral 38%, negative 35%,
positive 27%. Known data issues (notebook 01): 43 duplicate tweets in train, 9 with conflicting labels, 29 test tweets
that also occur in train (2.8%; notebooks 04 and 05 check the effect on the scores). Licence: CC BY 4.0.

## Notebooks (run in this order, on Google Colab)
| # | Notebook | What it does | Writes |
|---|----------|--------------|--------|
| 1 | `01_data_and_baseline` | exploration, cleaning, TF-IDF word and character baselines | `baseline_results.json`, `baseline_errors.csv` |
| 2 | `02_word2vec_bilstm` | Word2Vec + BiLSTM/BiGRU, 4 configurations x 3 seeds | `sequence_runs.json`, `sequence_results.csv` |
| 3 | `03_finetune_afroxlmr` | AfroXLMR-base/large and XLM-R base, 4 configurations x 3 seeds (GPU) | `finetune_runs.json`, `finetune_results.csv` |
| 3b | `03b_final_model_and_upload` | retrains the selected model, evaluates it, uploads to the Hub (GPU) | `final_metrics.json`, `test_predictions_final.csv`, `finetuned_errors.csv` |
| 3c | `03c_ablations` | class weights and casing ablations (GPU) | `ablation_results.csv` |
| 4 | `04_error_analysis` | error types, features, baseline comparison, confidence | `errors_tagged.csv`, `test_predictions.csv` |
| 5 | `05_evaluation_and_significance` | bootstrap confidence intervals, paired tests, per-class scores | `results_with_ci.csv`, `final_confusion_matrix.png` |

`results/` holds the result files of the runs reported here. Notebooks 02, 03, 04 and 05 read the result files of the earlier notebooks, so keep them in the same Colab folder or upload
them when asked.

## Results (test set, macro-F1)
Baselines are single runs; the sequence and fine-tuned models are mean ± std over seeds 42-44. Files are in `results/`.

| Model | Val macro-F1 | Test macro-F1 | Test accuracy |
|-------|--------------|---------------|---------------|
| TF-IDF word + LR | 0.579 | 0.567 | 0.562 |
| TF-IDF char + LR | 0.642 | 0.597 | 0.595 |
| BiLSTM + random embeddings | 0.549 ± 0.009 | 0.540 ± 0.008 | 0.539 ± 0.010 |
| BiLSTM + Word2Vec (frozen) | 0.602 ± 0.004 | 0.572 ± 0.007 | 0.571 ± 0.009 |
| BiLSTM + Word2Vec (fine-tuned) | 0.612 ± 0.006 | 0.580 ± 0.002 | 0.578 ± 0.001 |
| BiGRU + Word2Vec (fine-tuned) | 0.615 ± 0.007 | 0.575 ± 0.008 | 0.572 ± 0.008 |
| XLM-R base, lr 2e-5 | 0.558 ± 0.015 | 0.540 ± 0.019 | 0.554 ± 0.013 |
| AfroXLMR-base, lr 2e-5 | 0.669 ± 0.004 | 0.663 ± 0.008 | 0.661 ± 0.007 |
| AfroXLMR-base, lr 5e-5 | 0.661 ± 0.007 | 0.668 ± 0.006 | 0.665 ± 0.006 |
| AfroXLMR-large, lr 1e-5 (3 seeds) | 0.704 ± 0.008 | 0.707 ± 0.008 | 0.704 ± 0.009 |
| **Deployed model: AfroXLMR-large, seed 42, epoch 3** | 0.693 | **0.697** (95% CI 0.668-0.723) | 0.693 (0.664-0.720) |

Deployed model, per class (test): positive F1 0.740, neutral 0.653, negative 0.697 (`results/final_metrics.json`).
Seed 42 is the lowest of the three large-model seeds (0.697, 0.715, 0.710), so the three-seed mean is the experiment
result and 0.697 is the score of the uploaded model.

Ablations on AfroXLMR-base (notebook 03c): class weights 0.663 ± 0.008, no class weights 0.671 ± 0.004, lowercased
input 0.668 ± 0.012. None of the differences is larger than the seed-to-seed spread.

### Main findings
- Fine-tuned AfroXLMR-large beats the best baseline by 0.10 macro-F1 (paired bootstrap 95% CI [0.067, 0.130]).
- Large beats base in every seed (+0.034, +0.043, +0.056, all intervals exclude 0). Learning rates 2e-5 and 5e-5 are indistinguishable.
- AfroXLMR-base beats XLM-R base by about 0.12, but XLM-R was still improving at epoch 5, so part of that gap is slower convergence.
- The char baseline beats the word baseline by only 0.03 (CI [0.002, 0.059]): barely distinguishable from noise.
- The 29 test tweets that also occur in training do not inflate the scores (changes of at most 0.005).
- Neutral is the weakest class, and surface features (links, hashtags, emojis, replies) do not predict errors.

## Run the demo locally
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py   # loads Gkea/kinyarwanda-sentiment-afroxlmr-large from the Hugging Face Hub
```
The model weights are on the Hub and are not stored in this repository.

## Existing resources used
AfriSenti dataset (Muhammad et al., 2023), AfroXLMR (Alabi et al., 2022) and XLM-R (Conneau et al., 2020) checkpoints
from the Hugging Face Hub, `transformers`, `gensim`, `scikit-learn`, Streamlit. The notebooks, experiment design, error
analysis and app are my own work, written with AI assistance for coding and documentation (see the report).
