# Unified Cross-Mechanism Image Forgery Localization

This repository is the review version of the code release for the paper:

**Unified Cross-Mechanism Image Forgery Localization via Decoupled Sensitive and Invariant Representation Learning**

The current version is provided for review and verification purposes. It includes a demo script and the checkpoint selected from Stage 3 training. The complete version, including training code, full configuration files, data split scripts, evaluation scripts, and additional checkpoints, will be updated after the paper is accepted/published.

## Overview

This work studies unified pixel-level image forgery localization across heterogeneous forgery mechanisms, including conventional manipulations and generative forgeries. The proposed framework uses decoupled sensitive and invariant representation learning to preserve fine-grained localization cues while improving cross-mechanism generalization.

## Current Release

This review version includes:

- a demo script for testing a single input image;
- the Stage 3 checkpoint for verification (https://github.com/NeviNa-hub/Unified-Forgery-Localization/releases);
- this README file;
- a basic dependency file.

The demo is intended for inference verification only. Full training and evaluation scripts will be released in the complete version.

## Dataset Sources

Please download the datasets from their official sources:

- Columbia: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
- COVERAGE: https://github.com/wenbihan/coverage
- CASIA 1.0: https://github.com/namtpham/casia1groundtruth
- IMD2020: https://staff.utia.cas.cz/novozada/db/
- CocoGlide: https://github.com/grip-unina/TruFor
- DeepFakes: https://github.com/ondyari/FaceForensics
- DeepFakeDetection: originally introduced by Google Research at https://research.google/blog/contributing-data-to-deepfake-detection-research/ and accessed through the FaceForensics++ download interface at https://github.com/ondyari/FaceForensics/

## Quick Start

Install the basic dependencies:

```bash
pip install -r requirements.txt
```

Run the demo script with the released checkpoint:

```bash
python demo.py --input path/to/image.png --checkpoint path/to/stage3_checkpoint.pth --output path/to/output_mask.png
```

Please adjust the command according to the actual checkpoint and demo script names in this repository.



