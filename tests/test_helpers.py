import numpy as np
import pytest
from PIL import Image

from conftest import load_cells


def noise_image(seed=0, size=64):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def flat_image(value=128, size=64):
    return np.full((size, size, 3), value, dtype=np.uint8)


# --- convert: 12 metadata flags -> binary string -> one integer ---------------

def test_convert_joins_flags(helpers):
    assert helpers["convert"]([0, 1, 1, 0]) == "0110"


def test_convert_to_int_matches_notebook_usage(helpers):
    flags = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert int(helpers["convert"](flags), 2) == 1536


def test_convert_collapses_distinct_flag_sets_never(helpers):
    # 12 bits -> 4096 codes, so distinct flag vectors stay distinct (the info is kept, just not as 12 inputs).
    a = int(helpers["convert"]([1] + [0] * 11), 2)
    b = int(helpers["convert"]([0] * 11 + [1]), 2)
    assert a != b


# --- sharpness / colour measures ----------------------------------------------

def test_flat_image_has_zero_sharpness_and_colour(helpers):
    img = flat_image()
    assert helpers["image_colorfulness"](img) == 0
    assert helpers["tenengrad"](img) == 0
    assert helpers["variance_of_laplacian"](img) == 0


def test_noise_is_sharper_than_blur(helpers):
    import cv2
    img = noise_image()
    blurred = cv2.GaussianBlur(img, (9, 9), 3)
    assert helpers["tenengrad"](img) > helpers["tenengrad"](blurred)
    assert helpers["variance_of_laplacian"](img) > helpers["variance_of_laplacian"](blurred)


def test_grey_image_is_not_colourful_but_saturated_is(helpers):
    grey = flat_image(100)
    red = np.zeros((32, 32, 3), np.uint8)
    red[..., 0] = 255
    assert helpers["image_colorfulness"](grey) == 0
    assert helpers["image_colorfulness"](red) > 0


@pytest.mark.xfail(
    reason="BUG (review.md B2): image_colorfulness calls cv2.split and names the channels (B, G, R), but the "
    "notebook feeds it skimage.io.imread output, which is RGB. A pure-red RGB image should give "
    "0.3*sqrt(255**2 + 127.5**2) = 85.6 under the Hasler-Suesstrunk formula; the function gives 76.5.",
    strict=True,
)
def test_colorfulness_of_pure_red_rgb_image_matches_formula(helpers):
    red = np.zeros((32, 32, 3), np.uint8)
    red[..., 0] = 255  # RGB order, as skimage.io.imread returns
    expected = 0.3 * np.sqrt(255 ** 2 + 127.5 ** 2)
    assert helpers["image_colorfulness"](red) == pytest.approx(expected)


# --- calculate_brightness ------------------------------------------------------

def test_brightness_of_mid_grey_in_L_mode_is_about_half(helpers):
    img = Image.fromarray(flat_image(128)[..., 0], mode="L")
    assert helpers["calculate_brightness"](img) == pytest.approx(128 / 256)


def test_brightness_increases_with_grey_level(helpers):
    dark = Image.fromarray(flat_image(30)[..., 0], mode="L")
    light = Image.fromarray(flat_image(220)[..., 0], mode="L")
    assert helpers["calculate_brightness"](dark) < helpers["calculate_brightness"](light)


@pytest.mark.xfail(
    reason="BUG (review.md B3): on an RGB image PIL's histogram has 768 bins (R,G,B concatenated) and the "
    "function divides by len(histogram), so brightness mixes in the bin offsets and a dark grey RGB "
    "image does not score the same as the same grey in 'L' mode.",
    strict=True,
)
def test_brightness_rgb_equals_brightness_grey(helpers):
    # 128 is the one grey level where the two coincide by accident (0.5), so use a dark grey.
    grey_l = Image.fromarray(flat_image(30)[..., 0], mode="L")
    grey_rgb = Image.fromarray(flat_image(30), mode="RGB")
    assert helpers["calculate_brightness"](grey_rgb) == pytest.approx(helpers["calculate_brightness"](grey_l))


# --- get_data (cell 21) --------------------------------------------------------

def make_get_data_ns(train_csv):
    import pandas as pd
    ns = load_cells([7, 21], extra={"dir_train": "train/", "img_type": ".jpg", "train_csv": train_csv})
    return ns


def test_get_data_builds_paths_flag_codes_and_targets(train_csv):
    import pandas as pd
    ns = make_get_data_ns(train_csv)
    df_ic, df_ir = pd.DataFrame(columns=range(2)), pd.DataFrame(columns=range(2))
    ns["get_data"](train_csv, df_ic, df_ir)
    assert list(df_ic[0]) == ["train/aaa.jpg", "train/bbb.jpg", "train/ccc.jpg"]
    assert list(df_ic[1]) == [0b011000000000, 0b100000000001, 0]
    assert list(df_ir[1]) == [63, 42, 100]


@pytest.mark.xfail(
    reason="BUG (review.md B1): get_data reads the global train_csv for the flag code instead of its csv "
    "argument. Called with any other frame (e.g. a validation split), the paths and targets come "
    "from that frame but the flag codes come from train_csv.",
    strict=True,
)
def test_get_data_uses_its_csv_argument(train_csv):
    import pandas as pd
    other = train_csv.iloc[[2, 1, 0]].reset_index(drop=True)  # same rows, reversed
    ns = make_get_data_ns(train_csv)
    df_ic, df_ir = pd.DataFrame(columns=range(2)), pd.DataFrame(columns=range(2))
    ns["get_data"](other, df_ic, df_ir)
    assert list(df_ic[1]) == [0, 0b100000000001, 0b011000000000]


# --- cell 25: split -------------------------------------------------------------

def test_notebook_split_is_deterministic_80_20():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.DataFrame({"a": range(100)})
    tr1, te1 = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    tr2, te2 = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    assert len(tr1) == 80 and len(te1) == 20
    assert list(tr1.index) == list(tr2.index)
    assert set(tr1.index).isdisjoint(te1.index)
