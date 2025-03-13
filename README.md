# Kaggle Challenge Report: DNA Sequence Classification

**Authors:** Thomas Gravier, Thomas Loux

## Introduction

This report presents our approach to solving the DNA sequence classification challenge on Kaggle. The objective is to predict whether a region of a DNA sequence is a binding site for a specific transcription factor (TF). To achieve this, we experimented with several machine learning methods and different kernel functions tailored to biological sequences.

## Methodology

### Models Used

We tested three supervised classification models: **KNN** (K-Nearest Neighbors), **Ridge Regression**, and **SVM** (Support Vector Machine).

### Computation Optimization

- **KeOps** was used to accelerate the RBF kernel computation, reducing computation time.
- Attempted to use KeOps with `torch` for GPU acceleration, but it was incompatible with `scipy` because the solver requires interaction with `numpy.ndarray`.
- For SVM, we used `cvxpy`, which is 5 to 10 times faster than `scipy.optimize` for solving the quadratic problem.

### Kernels Used

We experimented with several kernels suitable for DNA sequence classification.

#### Spectrum Kernel

The **Spectrum Kernel** compares the occurrences of substrings (k-mers) of length \( k \) between two DNA sequences.

Given two sequences \( x \) and \( y \), and let \( \phi_k(x) \) be a vector of dimension \( 4^k \) representing the occurrences of all \( k \)-mers in \( x \). The kernel is defined as:

\[ K_{\text{spectrum}}(x, y) = \sum_{v \in \Sigma^k} \phi_k(x)_v \cdot \phi_k(y)_v \]

where \( \Sigma \) is the DNA alphabet (\{A, C, G, T\}).

The Spectrum Kernel is fast to compute and works well when \( k \) is appropriately chosen (\( k = 3 \) to \( 6 \) in our case). However, we did not use sparse structures.

#### Mismatch Kernel

The **Mismatch Kernel** is an extension of the Spectrum Kernel that tolerates a certain number of errors (mismatches) when comparing \( k \)-mers.

Let \( \phi_{k,m}(x) \) be a vector representing the occurrences of \( k \)-mers with up to \( m \) errors. The kernel is defined as:

\[ K_{\text{mismatch}}(x, y) = \sum_{v \in \Sigma^k} \sum_{w \in \mathcal{N}_m(v)} \phi_k(x)_w \cdot \phi_k(y)_w \]

where \( \mathcal{N}_m(v) \) is the set of \( k \)-mers that differ from \( v \) by at most \( m \) substitutions.

This kernel is more robust than the Spectrum Kernel but more computationally expensive.

#### Local Alignment Kernel

The **Local Alignment Kernel** measures the similarity between two sequences based on their optimal local alignments, making it particularly suitable for biological sequences.

It relies on an affinity function \( S(x, y) \) calculated by:

\[ K_{\text{LA}}(x, y) = \sum_{\text{alignments } A} e^{\lambda S(A)} \]

where \( S(A) \) is the local alignment score between \( x \) and \( y \), and \( \lambda \) is a control parameter.

This kernel seems to perform better according to papers by JP Vert but is computationally expensive, producing very high values (around \( 10^{20} \)). We observed that it was too slow for large datasets (limited to 100 samples in our tests). Despite using multiprocessing to parallelize computations, it took nearly 5 minutes, making it impractical due to quadratic cost.

### Optimization and Validation

- Implemented parallel processing via **multiprocessing** to speed up training.
- Conducted hyperparameter tuning with **GridSearchCV** and cross-validation (**CV**) for robust evaluation and to avoid overfitting. Accuracy values were quite sensitive to data partitioning.

## Results and Discussion

We compared the performance of the models and kernels in terms of accuracy and execution time.

| **Method**                          | **Accuracy (mean over KFold)** |
|-------------------------------------|---------------------------------|
| 3KNN + Spectrum Kernel (k=5)        | 55%                             |
| Ridge Regression + Spectrum Kernel (k=5) | 61%                         |
| SVM + Spectrum Kernel (k=5)         | **62%**                        |
| SVM + Mismatch Kernel (m=1, k=4)    | 58.6%                          |

The Ridge Regression or SVM models did not significantly affect performance. Computation times mainly depended on kernel calculations, taking up to a maximum of 10 minutes.

