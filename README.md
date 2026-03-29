# KeypointMatcher

## Overview

KeypointMatcher (`kp_matching`) is a Python package that provides a unified interface for keypoint matching (brute-force, kNN, FLANN) on top of OpenCV, including optional Lowe ratio test filtering and simple match visualization.  
It depends on [KeypointDetector](https://github.com/ry-yoshida-private/KeypointDetector) (`kp_detection`) for types such as `DetMatchResult` when combining detection and matching workflows.

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

## Example

After installing the package, import it from any directory. The snippet below uses random float descriptors; in practice you obtain descriptor matrices from a detector (for example via `kp_detection`).

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

`KPMatchMethod` (`BF`, `KNN`, `FLANN`) and dataclasses under `KPMatchingParameters` (`KPMatchCommonParameters`, `RatioTestParameters`, `FLANNParameters`) control matching behavior. Use `MatchingVisualizer` and `DrawMatchFlags` when drawing matches between two images.
