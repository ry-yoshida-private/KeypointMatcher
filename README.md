# KeypointMatcher

## Overview

KeypointMatcher (`kp_matching`) is a Python package that provides a unified interface for keypoint matching (brute-force, kNN, FLANN) on top of OpenCV, including optional Lowe ratio test filtering and simple match visualization.  
It depends on [KeypointDetector](https://github.com/ry-yoshida-private/KeypointDetector) (`kp_detection`) for types such as `KPDetectionResult` when combining detection and matching workflows.

For module-level layout and file roles, see [src/kp_matching/README.md](src/kp_matching/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode so changes to the source take effect immediately:

```bash
pip install -e .
```

`pyproject.toml` lists `numpy`, `opencv-contrib-python`, and `kp_detection` from Git; `pip install .` resolves those automatically.  
To install only the dependencies without the package:

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Examples

After installing the package, import it from any directory.

### 1. Match descriptor matrices directly

Use this when you already have two descriptor arrays (same layout OpenCV matchers expect: shape `(N, D)`, typically `float32`).

```python
import numpy as np

from kp_matching import KPMatchingParameters, KPMatchingProcessor

params = KPMatchingParameters.build_default_parameters()
processor = KPMatchingProcessor(params=params)

desc1 = np.random.randn(100, 128).astype(np.float32)
desc2 = np.random.randn(100, 128).astype(np.float32)

matches = processor.match(desc1, desc2)
# matches is a MatchResult; filter, iterate, or visualize as needed (see package README).
```

### 2. Pipeline from keypoint detection results

Use this when both images were processed with `kp_detection` and you have `KPDetectionResult` instances that include descriptors. `run_pipeline` checks that descriptors exist, runs the same matching path as `match`, and returns a `PairedDetectionResult` (detections plus `MatchResult`), which exposes matched coordinates and optional RANSAC filtering.

```python
from kp_detection import KPDetectionResult

from kp_matching import KPMatchingParameters, KPMatchingProcessor, PairedDetectionResult

params = KPMatchingParameters.build_default_parameters()
processor = KPMatchingProcessor(params=params)

# Build or load KPDetectionResult values from your detector (coordinates + descriptors).
query_det_result: KPDetectionResult = ...
gallery_det_result: KPDetectionResult = ...

paired: PairedDetectionResult = processor.run_pipeline(query_det_result, gallery_det_result)
```

`KPMatchMethod` (`BF`, `KNN`, `FLANN`) and dataclasses under `KPMatchingParameters` (`KPMatchCommonParameters`, `RatioTestParameters`, `FLANNParameters`) control matching behavior. Use `MatchingVisualizer` and `DrawMatchFlags` when drawing matches between two images.
