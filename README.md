# AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware  Large Language Model Serving

[![arXiv.2503.05096](https://img.shields.io/badge/arXiv-2503.05096-red)](https://arxiv.org/abs/2503.05096)

**Kaiyu Huang<sup>1 2</sup>, 
Hao Wu<sup>3</sup>, 
Zhubo Shi<sup>1</sup>, 
Han Zou<sup>1</sup>,
[Minchen Yu<sup>2 4 &#9993;</sup>](https://mincyu.github.io/),
Qingjiang Shi<sup>1 2 &#9993;</sup>**
<br>
<sup>1</sup>School of Computer Science and Technology, Tongji University
<sup>2</sup>Shenzhen Research Institute of Big Data, The Chinese University of Hong Kong, Shenzhen
<sup>3</sup>Huazhong University of Science and Technology
<sup>4</sup>School of Data Science, The Chinese University of Hong Kong, Shenzhen
<br>
&#9993;: Corresponding Authors

## 📍 Overview

AdaSpec: an efficient LLM inference system that **dynamically adjusts speculative strategies** according to real-time request loads and system configurations.  AdaSpec proposes a theoretical model to understand and predict the efficiency of speculative decoding across diverse scenarios. Additionally, it implements intelligent drafting and verification algorithms to guarantee optimal performance while **achieving high SLO attainment**. Experimental results on real-world LLM traces demonstrate that AdaSpec consistently meets SLOs and achieves substantial performance improvements, delivering up to **66% speedup** compared to state-of-the-art speculative inference systems.

![adaspec](./assets/overview.png)

## 🎉 News

- [2025-09] 🔥 This paper is accepted by ACM SoCC 2025.
- [2025-10] 🔔 The code of AdaSpec is released.

## 🛡 Setup Environment
Our implementation leverages the vLLM framework (version 0.6.3), with all experiments performed on NVIDIA L40 and A100 GPUs using CUDA 12.1.

```bash
apt update
apt install ccache

conda create -n adaspec python=3.10
conda activate adaspec

export MAX_JOBS=10
pip install -e .
```

## 💫 Getting Started


### 1. Preprocess the Spec-Bench Dataset

```bash
  cd benchmarks/adaspec/datasets
  python preprocess_specbench.py
```

### 2. Speedup Benchmark

```bash
  mkdir -p bench_results
  bash benchmarks/adaspec/scripts/7b.sh
  bash benchmarks/adaspec/scripts/33b.sh
  bash benchmarks/adaspec/scripts/70b.sh
```

## ⚙️ Implementation Details
The core implementation is primarily located in the following key files:
- vllm/spec_decode/batch_expansion.py
- vllm/spec_decode/draft_model_runner.py：Contains the main implementation of the AdaSpec.
- vllm/spec_decode/spec_decode_worker.py
- vllm/spec_decode/top1_proposer.py
- vllm/worker/worker.py

## 📝 Citation

If you use AdaSpec in your research, please cite the following paper.
```bibtex
@inproceedings{10.1145/3772052.3772239,
author = {Huang, Kaiyu and Wu, Hao and Shi, Zhubo and Zou, Han and Yu, Minchen and Shi, Qingjiang},
title = {AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving},
year = {2026},
isbn = {9798400722769},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3772052.3772239},
doi = {10.1145/3772052.3772239},
abstract = {Cloud-based Large Language Model (LLM) services often face challenges in achieving low inference latency and meeting Service Level Objectives (SLOs) under dynamic request patterns. Speculative decoding, which exploits lightweight models for drafting and LLMs for verification, has emerged as a compelling technique to accelerate LLM inference. However, existing speculative decoding solutions often fail to adapt to fluctuating workloads and dynamic system environments, resulting in impaired performance and SLO violations. In this paper, we introduce AdaSpec, an efficient LLM inference system that dynamically adjusts speculative strategies according to real-time request loads and system configurations. AdaSpec proposes a theoretical model to analyze and predict the efficiency of speculative strategies across diverse scenarios. Additionally, it implements intelligent drafting and verification algorithms to maximize performance while ensuring high SLO attainment. Experimental results on real-world LLM service traces demonstrate that AdaSpec consistently meets SLOs and achieves substantial performance improvements, delivering up to 66\% speedup compared to state-of-the-art speculative inference systems. The source code is publicly available at https://github.com/cerebellumking/AdaSpec},
booktitle = {Proceedings of the 2025 ACM Symposium on Cloud Computing},
pages = {361–374},
numpages = {14},
keywords = {large language models, machine learning inference},
location = {
},
series = {SoCC '25}
}
```

## 📚 Acknowledgements
This project is built upon the foundational work of the vLLM library and the contribution of Xiaoxuan Liu. 
Our codebase is a fork of [SmartSpec](https://github.com/LiuXiaoxuanPKU/vllm/tree/dsd), specifically from version `v0.6.3`. 
We extend our sincere gratitude to all the contributors of the vLLM project and to Xiaoxuan Liu for their outstanding implementation and for making their work publicly available.
