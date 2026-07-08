# SGH-GraphBNN: Stochastic-Gradient Hamiltonian Graph Bayesian Neural Network for IoT Intrusion Detection

This repository contains the reproducible implementation of **SGH-GraphBNN**, a graph-based Bayesian intrusion detection framework that combines a deterministic GraphSAGE encoder with a cached-embedding, low-rank Bayesian classifier trained using **Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)**. The implementation is designed for binary IoT intrusion classification on large-scale network-flow datasets and supports deterministic, calibrated, dropout-based, and Bayesian graph baselines within a single evaluation pipeline.

The main implementation is provided as a clean Jupyter notebook with no saved outputs. The notebook is intended to be executed from raw or processed dataset files and to regenerate all tables, figures, robustness results, diagnostic outputs, and statistical comparisons from the configured data.

---

## 1. Repository contents

```text
SGH_GraphBNN_Implementation_Reproducible.ipynb   # Main reproducible notebook
sgh_graphbnn_pipeline.py                         # Script version of the core pipeline
requirements.txt                                 # Python package requirements
config/sgh_graphbnn_config.yaml                  # Dataset paths and experiment settings
README.md                                        # Project documentation
outputs/                                         # Created at runtime for generated tables and figures
data/                                            # Optional local dataset directory
```

The notebook is the primary artifact. The Python script mirrors the major pipeline components for users who prefer command-line execution or modular inspection.

---

## 2. Method overview

SGH-GraphBNN is built around five connected stages:

1. **Dataset preprocessing**
   - Loads network-flow data.
   - Replaces missing and infinite values.
   - Encodes binary labels.
   - Selects numeric traffic-flow features.
   - Applies z-score normalization.

2. **Graph construction**
   - For **NF-ToN-IoT-v3**, the implementation supports a directed communication graph where network flows define edges between source and destination IPv4 endpoints.
   - For **CICIoT2023**, the implementation supports a flow-similarity IoT behaviour graph where flow records are represented as nodes and local similarity edges are constructed from scaled traffic-behaviour features.

3. **GraphSAGE representation learning**
   - Trains a deterministic GraphSAGE encoder.
   - Builds edge-level embeddings from source node embeddings, destination node embeddings, and edge attributes.
   - Trains a deterministic MLP edge classifier.

4. **Bayesian classifier inference with SGHMC**
   - Freezes the trained GraphSAGE encoder.
   - Caches edge embeddings to avoid repeated graph message passing during posterior sampling.
   - Samples a compact Bayesian classifier head using SGHMC.
   - Generates posterior predictive probabilities from retained SGHMC samples.

5. **Evaluation and supplementary reporting**
   - Computes predictive metrics, calibration metrics, uncertainty metrics, SGHMC diagnostics, OoD detection metrics, runtime, memory use, reliability curves, robustness tables, and pairwise statistical comparisons.

---

## 3. Model families implemented

The notebook supports the following model configurations:

| Model | Description |
|---|---|
| Base GNN (GraphSAGE) | Deterministic GraphSAGE encoder with an MLP edge classifier. |
| Temperature-Scaled GNN | Base GraphSAGE model calibrated using temperature scaling. |
| GNN + MC Dropout | GraphSAGE model evaluated with dropout active during inference. |
| SGH-GraphBNN | GraphSAGE encoder with cached embeddings and a Bayesian SGHMC classifier head. |

The proposed SGH-GraphBNN model is labelled consistently as **SGH-GraphBNN** in generated outputs.

---

## 4. Dataset requirements

### 4.1 NF-ToN-IoT-v3

The NF-ToN-IoT-v3 experiment expects a CSV file containing network-flow records with at least:

- A binary or attack/benign label column.
- Source and destination IPv4 endpoint columns for communication graph construction.
- Numeric traffic-flow features.

Typical column names can be configured in `config/sgh_graphbnn_config.yaml`.

Recommended fields:

```text
IPV4_SRC_ADDR
IPV4_DST_ADDR
Label
```

If the source and destination columns are unavailable, the notebook can fall back to a similarity-graph construction strategy, but the directed communication graph is the preferred formulation for NF-ToN-IoT-v3.

### 4.2 CICIoT2023

The CICIoT2023 experiment expects processed CSV flow files or a consolidated processed CSV file. The notebook supports:

- Binary benign/attack label mapping.
- Removal of timestamp-like fields before modelling.
- Numeric traffic-behaviour feature selection.
- Flow-similarity graph construction using k-nearest neighbours.

The workflow is memory-aware and allows chunked loading, row limits for pilot runs, and configurable graph construction sizes.

### 4.3 Data placement

Recommended local structure:

```text
data/
  NF-ToN-IoT-v3/
    NF-ToN-IoT-v3.csv
  CICIoT2023/
    *.csv
```

The exact paths are controlled from:

```text
config/sgh_graphbnn_config.yaml
```

---

## 5. Installation

### 5.1 Create an environment

```bash
python -m venv sgh_graphbnn_env
source sgh_graphbnn_env/bin/activate
```

On Windows:

```bash
python -m venv sgh_graphbnn_env
sgh_graphbnn_env\Scripts\activate
```

### 5.2 Install requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.3 PyTorch and PyTorch Geometric

PyTorch and PyTorch Geometric installations depend on the CUDA version available on the machine. If the standard installation fails, install the correct PyTorch build first from the official PyTorch installation instructions and then install the matching PyTorch Geometric packages.

For CPU-only testing, the notebook can run smaller experiments by reducing dataset size and graph size in the configuration file.

---

## 6. Configuration

All major experiment settings are controlled in:

```text
config/sgh_graphbnn_config.yaml
```

Key configuration groups include:

```yaml
data:
  nf_ton_iot_path: data/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv
  ciciot2023_dir: data/CICIoT2023
  label_column: Label
  binary_positive_labels: [Attack, attack, 1]

split:
  train_size: 0.80
  validation_size: 0.08
  test_size: 0.12
  seeds: [42, 123, 2024, 3407, 777]

graph:
  top_numeric_features: 32
  append_edge_l2_norm: true
  knn_neighbors: 10

model:
  hidden_dim: 128
  graph_layers: 2
  encoder_dropout: 0.10
  classifier_dropout: 0.25

training:
  optimizer: AdamW
  learning_rate: 0.002
  weight_decay: 0.00001
  label_smoothing: 0.01
  max_epochs: 100
  early_stopping_patience: 5

sghmc:
  chains: 2
  burn_in: 300
  retained_samples_per_chain: 200
  total_iterations_per_chain: 500
  step_size: 0.0184
  friction: 0.051
  noise_scale: 0.0100
  likelihood_subset_size: 60000
```

These values can be adjusted for local compute availability.

---

## 7. Running the notebook

Open the notebook:

```text
SGH_GraphBNN_Implementation_Reproducible.ipynb
```

Run the cells sequentially. The notebook is organized into the following sections:

1. Environment setup and imports
2. Configuration loading
3. Reproducibility utilities
4. Dataset loading
5. Preprocessing and feature selection
6. Graph construction
7. GraphSAGE model definition
8. Deterministic training loop
9. Temperature scaling baseline
10. MC Dropout inference
11. Cached edge embedding extraction
12. Low-rank Bayesian classifier head
13. SGHMC sampler
14. Posterior predictive inference
15. Calibration and uncertainty metrics
16. Reliability-curve computation
17. OoD and misclassification-detection analysis
18. Runtime and memory tracking
19. Five-seed robustness evaluation
20. Statistical comparison of SGH-GraphBNN against baselines
21. Supplementary table export
22. Supplementary figure export
23. Output manifest generation

The notebook intentionally contains no saved outputs. Tables and figures are generated when the notebook is executed.

---

## 8. Running the script

The command-line pipeline can be executed with:

```bash
python sgh_graphbnn_pipeline.py --config config/sgh_graphbnn_config.yaml
```

Optional arguments may include:

```bash
python sgh_graphbnn_pipeline.py \
  --config config/sgh_graphbnn_config.yaml \
  --dataset nf_ton_iot \
  --seed 42 \
  --output-dir outputs
```

The script is useful for running repeatable experiments outside Jupyter.

---

## 9. Main outputs generated at runtime

The notebook writes outputs to:

```text
outputs/
  tables/
  figures/
  diagnostics/
  predictions/
  logs/
```

Typical generated tables include:

- Dataset summary table
- Feature summary table
- Graph construction summary table
- Classification performance table
- Calibration and uncertainty table
- SGHMC parameter posterior summary table
- SGHMC chain-level diagnostics table
- Subset-size convergence table
- Runtime and memory table
- OoD detection table
- Five-seed robustness table
- Pairwise statistical comparison table
- Figure manifest table

Typical generated figures include:

- SGH-GraphBNN workflow diagram
- Dataset scale and graph-construction figure
- Cached-embedding efficiency diagram
- Embedding visualization
- SGHMC trace plots
- Posterior density plots
- Posterior HDI coefficient plot
- SGHMC diagnostic plot
- Subset-size convergence plot
- Classification-performance comparison figures
- Calibration-error comparison figures
- Prediction-entropy figures
- Joint reliability diagram
- Runtime and memory comparison figure
- OoD detection figure
- Five-seed robustness figures
- ROC and precision-recall curves
- Entropy-distribution plots
- Metric-wise ranking heatmap

---

## 10. Evaluation metrics

### 10.1 Classification metrics

The notebook computes:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- AUROC
- AUPR

### 10.2 Calibration metrics

The notebook computes:

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Negative log-likelihood
- Brier score
- Reliability-bin statistics

### 10.3 Uncertainty metrics

The notebook computes:

- Predictive entropy
- Mean prediction entropy
- Standard deviation of prediction entropy
- Posterior predictive variance
- Confidence distribution summaries

### 10.4 SGHMC diagnostics

The notebook computes:

- R-hat
- Bulk ESS
- Tail ESS
- MCSE mean
- MCSE standard deviation
- Lag-1 autocorrelation
- Energy drift
- Parameter update norm
- Gradient-noise variance estimate
- Momentum norm

### 10.5 OoD and misclassification detection metrics

The notebook supports uncertainty-based detection analysis using:

- Misclassification AUROC
- Misclassification AUPR
- FPR@95%TPR
- Entropy-based confidence separation

### 10.6 Runtime and memory metrics

The notebook records:

- Graph preprocessing time
- Encoder training time
- Cached embedding extraction time
- SGHMC sampling time
- Posterior prediction time
- Total runtime
- Peak memory
- Posterior sample memory
- Throughput indicators

---

## 11. Statistical comparison

The notebook includes a full statistical comparison block for evaluating SGH-GraphBNN against baselines across repeated seeds.

Implemented comparisons include:

- SGH-GraphBNN vs Base GNN (GraphSAGE)
- SGH-GraphBNN vs Temperature-Scaled GNN
- SGH-GraphBNN vs GNN + MC Dropout

For each metric and comparator, the notebook computes:

- Mean ± standard deviation
- Paired Wilcoxon signed-rank p-value
- Holm-adjusted p-value
- Cliff's delta effect size
- Direction of improvement
- Metric-level interpretation field

The statistical comparison is generated from the repeated-seed results produced during notebook execution. It does not require manually entered manuscript values.

---

## 12. Reproducibility controls

The implementation sets random seeds for:

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA

It also configures deterministic execution options where supported by the installed backend. Because GPU kernels and sparse graph operations can introduce small numerical differences across hardware and library versions, exact bitwise identity is not guaranteed across all machines. However, the repeated-seed protocol is designed to produce stable aggregate estimates.

Recommended seed set:

```text
42, 123, 2024, 3407, 777
```

---

## 13. Computational design

SGH-GraphBNN is designed to make Bayesian graph inference computationally practical for large-scale IoT intrusion detection. The core computational design choices are:

1. **Encoder freezing**
   - The GraphSAGE encoder is trained once and then frozen before SGHMC sampling.

2. **Cached edge embeddings**
   - Edge embeddings are extracted once and reused by the Bayesian classifier head.

3. **Low-rank Bayesian classifier**
   - Bayesian sampling is restricted to a compact classifier head rather than the full graph encoder.

4. **Mini-batch SGHMC**
   - Posterior sampling uses stochastic mini-batches rather than full-batch likelihood evaluations.

5. **Posterior sample compression**
   - Retained posterior samples can be stored using mixed precision to reduce memory consumption.

6. **Early posterior stabilization checks**
   - Sampling diagnostics are monitored to assess whether retained posterior samples are stable enough for posterior prediction.

This design avoids repeated full-graph propagation during posterior sampling and supports Bayesian uncertainty estimation with lower computational overhead than full graph-level posterior sampling.

---

## 14. Hardware guidance

Recommended hardware for full-scale runs:

- NVIDIA GPU with CUDA support
- At least 16 GB GPU memory for large subgraphs
- At least 32 GB system RAM for large CSV processing

For smaller machines:

- Reduce `row_limit` or enable chunked loading.
- Reduce SGHMC likelihood subset size.
- Reduce graph neighbour count.
- Run one dataset at a time.
- Use CPU mode for smoke tests.
- Reduce retained posterior samples.

Suggested smoke-test settings:

```yaml
data:
  row_limit: 50000
sghmc:
  likelihood_subset_size: 12000
  total_iterations_per_chain: 100
  burn_in: 40
  retained_samples_per_chain: 30
graph:
  knn_neighbors: 5
```

---

## 15. Expected workflow for generating supporting data

A recommended full execution workflow is:

1. Update dataset paths in `config/sgh_graphbnn_config.yaml`.
2. Run dataset validation cells.
3. Run preprocessing cells.
4. Build dataset-specific graphs.
5. Train the deterministic GraphSAGE encoder.
6. Evaluate Base GNN, Temperature-Scaled GNN, and MC Dropout baselines.
7. Cache edge embeddings.
8. Run SGHMC posterior sampling.
9. Generate posterior predictions.
10. Compute all metrics.
11. Run five-seed robustness evaluation.
12. Run statistical comparison.
13. Export tables and figures.
14. Review `outputs/manifest.csv` for generated supporting files.

---

## 16. Troubleshooting

### CUDA out-of-memory error

Reduce one or more of the following:

- Training batch size
- SGHMC mini-batch size
- Likelihood subset size
- Number of retained posterior samples
- k-nearest-neighbour graph size
- Number of rows loaded from the dataset

### Slow k-nearest-neighbour graph construction

Use one or more of the following:

- Enable approximate nearest neighbours if available.
- Reduce the number of loaded CICIoT2023 rows for exploratory runs.
- Reduce `knn_neighbors`.
- Precompute and cache the similarity graph.

### Missing IP columns in NF-ToN-IoT-v3

Check the configured source and destination column names. If the CSV uses different names, update the configuration file. If endpoint columns are unavailable, use the similarity-graph option.

### Label column not found

Update:

```yaml
data:
  label_column: Label
```

or provide a label mapping function in the preprocessing cell.

### AUROC or AUPR cannot be computed

This occurs when a split contains only one class. Use stratified splitting, verify label mapping, and check whether row filtering removed one class.

### PyTorch Geometric installation issues

Install the PyTorch version first, then install matching PyTorch Geometric wheels. Check CUDA compatibility before installing GPU builds.

---

## 17. Extending the notebook

The notebook is structured to support additional supplementary outputs. Possible extensions include:

- Additional calibration bin counts
- Adaptive calibration error
- Class-specific precision and recall
- Confusion matrices by dataset
- Confidence histogram by class
- Entropy distribution by correctness
- Embedding visualizations by dataset
- Runtime scaling by likelihood subset size
- Memory scaling by posterior sample count
- SGHMC sensitivity analysis for step size and friction
- Ablation of cached embeddings
- Ablation of low-rank classifier factorization
- Ablation of SGHMC posterior sample count
- Cross-dataset generalization tests

---

## 18. Output naming convention

Generated files follow a consistent naming convention:

```text
<dataset>_<experiment>_<metric_or_figure>.csv
<dataset>_<experiment>_<metric_or_figure>.png
combined_<experiment>_<metric_or_figure>.csv
combined_<experiment>_<metric_or_figure>.png
```

Examples:

```text
nf_ton_iot_classification_metrics.csv
ciciot2023_calibration_metrics.csv
combined_five_seed_robustness.csv
combined_statistical_comparison.csv
combined_reliability_diagram.png
sghmc_parameter_diagnostics.csv
```

---

## 19. Documentation standards

The notebook and script use descriptive section headers and comments to make the workflow auditable. Comments explain the purpose of each block, the expected inputs, and the generated outputs. The implementation is intended to be read as a supplementary research artifact as well as executed as a reproducible experiment.

---

## 20. License and data-use note

This repository does not redistribute the original datasets. Users should obtain NF-ToN-IoT-v3 and CICIoT2023 from their official sources and comply with the dataset providers' usage terms.

The code is intended for academic and research use in intrusion detection, graph learning, Bayesian uncertainty estimation, and reproducible machine-learning evaluation.

---

## 21. Citation

When using this implementation, cite the associated SGH-GraphBNN manuscript and acknowledge the dataset sources used in the experiment.

```bibtex
@article{SGHGraphBNN,
  title   = {SGH-GraphBNN: Stochastic-Gradient Hamiltonian Graph Bayesian Neural Network for Uncertainty-Aware IoT Intrusion Detection},
  author  = {Osei-Wusu, Franco; Appiah, Obed; Mensah, Patrick K.; Nimbe, Peter; Donkoh, Elvis K.},
  year    = {2026},
  note    = {Manuscript and supplementary implementation}
}
```

---

## 22. Contact

For questions about the implementation, configuration, or supplementary outputs, contact the manuscript author or corresponding maintainer.
