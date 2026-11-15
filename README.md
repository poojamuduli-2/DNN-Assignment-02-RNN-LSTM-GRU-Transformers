# DNN Programming Assignment 2 — 2026AK05702

**Student:** Pooja Muduli  
**Problems:** 1 — RNN/LSTM/GRU; 2 — Transformer

## Chosen settings (instructor clarification)

* **Problem 1:** hidden size **h = 64** for RNN, LSTM and GRU; three seeds **42, 123, 2026**. HAR: 8 epochs; adding task: 5 epochs, lengths 50/100/200/400. LSTM/GRU hidden-size sweep: 16/32/64/128. Student ID ends in 2, so the assigned ablation is RNN gradient clipping (norm 1 versus none).
* **Problem 2:** width **d = 128** (divisible by 1, 4 and 8 heads); main model: 4 heads, 4 language-model layers, context 128; HAR: 8 epochs with the same three seeds. Shakespeare: 1,000 training steps, batch size 32. Head-count and mask ablations: 300 steps per condition.
* **Datasets:** UCI HAR (four training subjects held out per seed; normalization fitted to training subjects only), generated adding sequences, and Tiny Shakespeare (first 90% training; last 10% validation).

## Environment

Python 3.10+, PyTorch 2.x, NumPy, pandas, Matplotlib, scikit-learn and IPython (see `requirements.txt`). **Google Colab with an NVIDIA T4 GPU** is recommended; notebooks require internet access to download datasets. The saved outputs identify **PyTorch 2.14.1+cu130 / Tesla T4**; other package versions were not recorded and should be captured from the execution environment.

## Run instructions

1. Upload `problem1/problem1.ipynb` to Colab. Select **Runtime → Change runtime type → T4 GPU**, then **Runtime → Restart session and run all**. Save the executed notebook, `problem1/figures/`, and its results.
2. In a fresh Colab session, run `problem2/problem2.ipynb` the same way. Save the executed notebook, `problem2/figures/`, `problem2/generated\_samples.txt`, and its results. Problem 1 HAR comparison rows are embedded in Problem 2, so both notebooks need not share a runtime.
3. Combine experiment rows into the root `results.csv` with the assignment's exact 12 columns. Include the figures and `report.pdf` in the final ZIP; exclude datasets and checkpoints.

## Total T4 runtime

**Measured total: 6 min 52 sec on NVIDIA T4** (Problem 1: 2 min 40 sec; Problem 2: 4 min 12 sec). Timings supplied from complete notebook executions.

## Known limitations / issues

Both notebooks have saved experiment outputs; no unresolved execution failure is documented. The Problem 1 GRU cell equivalence check uses a compatible single-bias configuration. The 256/512-token language-model benchmarks extend embeddings beyond the trained 128-token context and assess computational scaling, **not** prediction quality. GPU results may vary slightly. 

