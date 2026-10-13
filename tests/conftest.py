"""Load helper functions straight from the notebook, so the tests exercise the real code.

The notebook is Colab-only and cannot be imported, so we exec just the cells we need.
"""
import json
import pathlib

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Copy_o-checkpoint.ipynb"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def cell_source(index):
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    return "".join(nb["cells"][index]["source"])


def load_cells(indices, extra=None):
    ns = {"np": np, "cv2": cv2, "pd": pd}
    ns.update(extra or {})
    for i in indices:
        exec(compile(cell_source(i), f"notebook-cell-{i}", "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def helpers():
    # 7 convert, 10 image_colorfulness, 11 calculate_brightness, 12 tenengrad, 13 variance_of_laplacian
    return load_cells([7, 10, 11, 12, 13])


@pytest.fixture
def train_csv():
    return pd.read_csv(FIXTURES / "mini_train.csv")
