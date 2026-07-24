from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import polars as pl

try:
    from scipy.stats import hypergeom as scipy_hypergeom
except Exception:  # pragma: no cover - scipy is optional.
    scipy_hypergeom = None


@dataclass(frozen=True)
class ScoreParams:
    p_B: float = 0.95
    p_G: float = 0.20
    alpha_B: float = 0.7
    alpha_G: float = 0.4
    k_N: float = 5.0
    virtual_good_h: int = 3
    eps: float = 1e-12


DISPLAY_COLUMNS = [
    "step_seq",
    "root_lot_count",
    "root_lot_ids",
    "input_rows",
    "N_real",
    "n_B_real",
    "n_G_real",
    "m_real",
    "N_eff",
    "B_eff",
    "m_eff",
    "virtual_good_h",
    "n_B1",
    "n_B0",
    "n_G1",
    "n_G0",
    "mean_likelihood",
    "quality_real",
    "p_tail_eff",
    "evidence_eff",
    "r_real",
    "score",
    "score_exact_alt",
]


SINGLE_STEP_LOT_REQUIRED_COLUMNS = {"root_lot_id", "wafer_id", "good_bad", "y"}

SINGLE_STEP_LOT_OUTPUT_COLUMNS = [
    "step_seq",
    "root_lot_count",
    "root_lot_ids",
    "input_rows",
    "N_real",
    "N_eff",
    "B_eff",
    "m_eff",
    "n_B1",
    "n_B0",
    "n_G1",
    "n_G0",
    "quality_real",
    "p_tail_eff",
    "evidence_eff",
    "r_real",
    "score",
]


def _safe_prob(p: float, eps: float) -> float:
    """Clamp a probability away from 0 and 1 for stable log/ratio math."""
    return min(max(float(p), eps), 1.0 - eps)


def log_comb(n: int, k: int) -> float:
    """Return log(C(n, k)) using lgamma to avoid integer overflow."""
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return -math.inf
    max_val = max(vals)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in vals))


def hypergeom_tail_pvalue(N: int, n_B: int, m: int, x_obs: int) -> float:
    """
    Compute P(X >= x_obs), X ~ Hypergeometric(N, n_B, m).

    Degenerate cases return 1.0 so evidence becomes zero. Good-class contrast
    may be supplied by an upstream virtual-good prior before calling this function.
    """
    N = int(N)
    n_B = int(n_B)
    m = int(m)
    x_obs = int(x_obs)
    n_G = N - n_B

    if N <= 0 or n_B <= 0 or n_G < 0 or m <= 0 or m >= N:
        return 1.0
    if not (0 <= n_B <= N and 0 <= m <= N):
        raise ValueError(f"Invalid hypergeometric parameters: N={N}, n_B={n_B}, m={m}")

    lower = max(0, m - n_G)
    upper = min(n_B, m)
    if x_obs <= lower:
        return 1.0
    if x_obs > upper:
        return 0.0

    if scipy_hypergeom is not None:
        # scipy sf(k) = P(X > k), so sf(x_obs - 1) = P(X >= x_obs).
        p_tail = float(scipy_hypergeom.sf(x_obs - 1, N, n_B, m))
        return min(max(p_tail, 0.0), 1.0)

    log_den = log_comb(N, m)
    log_terms = (
        log_comb(n_B, x) + log_comb(n_G, m - x) - log_den
        for x in range(max(x_obs, lower), upper + 1)
    )
    p_tail = math.exp(_logsumexp(log_terms))
    return min(max(p_tail, 0.0), 1.0)


def _validate_counts(*counts: int) -> None:
    for count in counts:
        if int(count) != count or int(count) < 0:
            raise ValueError(f"Counts must be non-negative integers: {counts}")


def _validate_params(params: ScoreParams) -> None:
    if int(params.virtual_good_h) != params.virtual_good_h or params.virtual_good_h < 0:
        raise ValueError(f"virtual_good_h must be a non-negative integer: {params.virtual_good_h}")
    if params.k_N < 0:
        raise ValueError(f"k_N must be non-negative: {params.k_N}")


def score_process_counts(
    n_B1: int,
    n_B0: int,
    n_G1: int,
    n_G0: int,
    params: Optional[ScoreParams] = None,
) -> Dict[str, float]:
    """Score one process step from its 2x2 good_bad/context count table."""
    params = params or ScoreParams()
    _validate_params(params)
    _validate_counts(n_B1, n_B0, n_G1, n_G0)

    n_B1 = int(n_B1)
    n_B0 = int(n_B0)
    n_G1 = int(n_G1)
    n_G0 = int(n_G0)

    p_B = _safe_prob(params.p_B, params.eps)
    p_G = _safe_prob(params.p_G, params.eps)
    virtual_good_h = int(params.virtual_good_h)

    n_B_real = n_B1 + n_B0
    n_G_real = n_G1 + n_G0
    N_real = n_B_real + n_G_real
    m_real = n_B1 + n_G1

    n_B1_eff = n_B1
    n_B0_eff = n_B0
    n_G1_eff = n_G1
    n_G0_eff = n_G0 + virtual_good_h
    N_eff = n_B1_eff + n_B0_eff + n_G1_eff + n_G0_eff
    B_eff = n_B1_eff + n_B0_eff
    m_eff = n_B1_eff + n_G1_eff
    x_obs = n_B1_eff

    w_B0 = math.log(p_B / (1.0 - p_B))
    w_G1 = math.log((1.0 - p_G) / p_G)

    if N_real == 0:
        log_likelihood = 0.0
        mean_likelihood = 0.0
        ideal_log_likelihood = 0.0
        r_real = 0.0
    else:
        log_likelihood = (
            n_B1 * math.log(p_B)
            + n_B0 * math.log(1.0 - p_B)
            + n_G1 * math.log(p_G)
            + n_G0 * math.log(1.0 - p_G)
        )
        mean_likelihood = math.exp(log_likelihood / N_real)
        ideal_log_likelihood = n_B_real * math.log(p_B) + n_G_real * math.log(1.0 - p_G)
        r_real = N_real / (N_real + params.k_N) if (N_real + params.k_N) > 0 else 0.0

    penalty_raw = n_B0 * w_B0 + n_G1 * w_G1
    bad_miss_rate = n_B0 / n_B_real if n_B_real > 0 else 0.0
    good_false_context_rate = n_G1 / n_G_real if n_G_real > 0 else 0.0
    if n_B_real == 0:
        J_bal = 0.0
    elif n_G_real > 0:
        J_bal = (
            params.alpha_B * w_B0 * bad_miss_rate
            + params.alpha_G * w_G1 * good_false_context_rate
        )
    else:
        J_bal = params.alpha_B * w_B0 * bad_miss_rate
    quality_real = math.exp(-J_bal)

    p_tail_eff = hypergeom_tail_pvalue(N=N_eff, n_B=B_eff, m=m_eff, x_obs=x_obs)
    evidence_eff = 0.0 if p_tail_eff >= 1.0 else -math.log(max(p_tail_eff, params.eps))
    score = 0.0 if n_B_real == 0 else r_real * quality_real * evidence_eff

    log_count_prob_alt = (
        log_comb(n_B_real, n_B1)
        + n_B1 * math.log(p_B)
        + n_B0 * math.log(1.0 - p_B)
        + log_comb(n_G_real, n_G1)
        + n_G1 * math.log(p_G)
        + n_G0 * math.log(1.0 - p_G)
    )
    count_prob_alt = float(math.exp(log_count_prob_alt)) if log_count_prob_alt > math.log(np.finfo(float).tiny) else 0.0
    score_exact_alt = r_real * quality_real * (-math.log(max(count_prob_alt, params.eps)))

    return {
        "N": N_real,
        "n_B": n_B_real,
        "n_G": n_G_real,
        "m": m_real,
        "N_real": N_real,
        "n_B_real": n_B_real,
        "n_G_real": n_G_real,
        "m_real": m_real,
        "n_B1_eff": n_B1_eff,
        "n_B0_eff": n_B0_eff,
        "n_G1_eff": n_G1_eff,
        "n_G0_eff": n_G0_eff,
        "N_eff": N_eff,
        "B_eff": B_eff,
        "m_eff": m_eff,
        "x_obs": x_obs,
        "n_B1": n_B1,
        "n_B0": n_B0,
        "n_G1": n_G1,
        "n_G0": n_G0,
        "p_B": p_B,
        "p_G": p_G,
        "virtual_good_h": virtual_good_h,
        "w_B0": w_B0,
        "w_G1": w_G1,
        "log_likelihood": log_likelihood,
        "mean_likelihood": mean_likelihood,
        "ideal_log_likelihood": ideal_log_likelihood,
        "penalty_raw": penalty_raw,
        "J_bal": J_bal,
        "quality": quality_real,
        "quality_real": quality_real,
        "p_tail": p_tail_eff,
        "p_tail_eff": p_tail_eff,
        "evidence": evidence_eff,
        "evidence_eff": evidence_eff,
        "r_N": r_real,
        "r_real": r_real,
        "score": score,
        "log_count_prob_alt": log_count_prob_alt,
        "count_prob_alt": count_prob_alt,
        "score_exact_alt": score_exact_alt,
    }


def _normalize_good_bad_expr() -> pl.Expr:
    """Return a Polars expression that normalizes good_bad to G or B."""
    return (
        pl.col("good_bad")
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(
            {"good": "G", "g": "G", "bad": "B", "b": "B"},
            default=None,
            return_dtype=pl.String,
        )
        .alias("_good_bad_norm")
    )


def _normalize_y_expr() -> pl.Expr:
    """Return a non-strict numeric cast for validating context y."""
    return pl.col("y").cast(pl.Float64, strict=False).alias("_y_number")


def _with_normalized_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_normalize_good_bad_expr(), _normalize_y_expr()).with_columns(
        pl.col("_y_number").cast(pl.Int64).alias("_y_norm")
    )


def _to_polars_dataframe(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialize Polars input as a cloned DataFrame."""
    if isinstance(data, pl.DataFrame):
        return data.clone()
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    raise TypeError("Input must be a polars DataFrame or LazyFrame.")


def _ordered_root_lot_ids(df: pl.DataFrame) -> List[str]:
    """Return distinct root lot IDs in their first-seen order."""
    return (
        df.get_column("root_lot_id")
        .cast(pl.String)
        .unique(maintain_order=True)
        .to_list()
    )


def _display_invalid_values(df: pl.DataFrame, column: str) -> List[str]:
    return [
        str(value)
        for value in df.get_column(column).unique(maintain_order=True).to_list()
    ]


def _validate_wafer_identity_rows(
    df: pl.DataFrame,
    identity_columns: List[str],
    max_wafers: int,
) -> None:
    """Validate per-lot wafer IDs and reject duplicate wafer identities."""
    if max_wafers < 1:
        raise ValueError("max_wafers must be at least 1.")

    checked = df.with_columns(
        pl.col("wafer_id").cast(pl.Float64, strict=False).alias("_wafer_id_number")
    )
    invalid_numeric = checked.filter(
        pl.col("_wafer_id_number").is_null()
        | pl.col("_wafer_id_number").is_nan()
    )
    if invalid_numeric.height:
        invalid = _display_invalid_values(invalid_numeric, "wafer_id")
        raise ValueError(
            f"wafer_id must be numeric values from 1 to {max_wafers}: {invalid}"
        )

    non_integer = checked.filter((pl.col("_wafer_id_number") % 1) != 0)
    if non_integer.height:
        raise ValueError("wafer_id must be integer-valued.")

    invalid_range = checked.filter(
        (pl.col("_wafer_id_number") < 1)
        | (pl.col("_wafer_id_number") > max_wafers)
    )
    if invalid_range.height:
        invalid = _display_invalid_values(invalid_range, "wafer_id")
        raise ValueError(
            f"wafer_id must be in 1..{max_wafers} within each root lot: {invalid}"
        )

    duplicated = (
        df.group_by(identity_columns, maintain_order=True)
        .agg(pl.len().alias("_row_count"))
        .filter(pl.col("_row_count") > 1)
        .select(identity_columns)
    )
    if duplicated.height:
        raise ValueError(
            f"Duplicate wafer identity rows found: {duplicated.to_dicts()}"
        )


def _validate_good_bad_and_y(df: pl.DataFrame) -> None:
    checked = df.with_columns(_normalize_good_bad_expr(), _normalize_y_expr())

    invalid_good_bad = checked.filter(pl.col("_good_bad_norm").is_null())
    if invalid_good_bad.height:
        invalid = _display_invalid_values(invalid_good_bad, "good_bad")
        raise ValueError(f"Invalid good_bad values: {invalid}")

    invalid_y = checked.filter(
        pl.col("_y_number").is_null()
        | pl.col("_y_number").is_nan()
        | ((pl.col("_y_number") % 1) != 0)
        | ~pl.col("_y_number").is_in([0.0, 1.0])
    )
    if invalid_y.height:
        invalid = _display_invalid_values(invalid_y, "y")
        raise ValueError(f"Invalid y values: {invalid}")


def _validate_input_dataframe(df: pl.DataFrame, max_wafers: int = 25) -> None:
    required = {"step_seq", "root_lot_id", "wafer_id", "good_bad", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["step_seq", "root_lot_id", "wafer_id"]:
        if df.get_column(column).null_count():
            raise ValueError(f"{column} must not contain missing values.")

    _validate_wafer_identity_rows(
        df,
        identity_columns=["step_seq", "root_lot_id", "wafer_id"],
        max_wafers=max_wafers,
    )
    _validate_good_bad_and_y(df)


def _validate_single_step_lot_dataframe(
    df: pl.DataFrame,
    max_wafers: int,
) -> List[str]:
    missing = SINGLE_STEP_LOT_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.is_empty():
        raise ValueError("Input DataFrame must contain at least one wafer row.")
    if df.get_column("root_lot_id").null_count():
        raise ValueError("root_lot_id must not contain missing values.")

    root_lot_ids = _ordered_root_lot_ids(df)
    _validate_wafer_identity_rows(
        df,
        identity_columns=["root_lot_id", "wafer_id"],
        max_wafers=max_wafers,
    )
    _validate_good_bad_and_y(df)
    return root_lot_ids


def _count_good_bad_context(df: pl.DataFrame) -> Dict[str, int]:
    normalized = _with_normalized_columns(df)
    row = normalized.select(
        (
            (pl.col("_good_bad_norm") == "B") & (pl.col("_y_norm") == 1)
        ).sum().alias("n_B1"),
        (
            (pl.col("_good_bad_norm") == "B") & (pl.col("_y_norm") == 0)
        ).sum().alias("n_B0"),
        (
            (pl.col("_good_bad_norm") == "G") & (pl.col("_y_norm") == 1)
        ).sum().alias("n_G1"),
        (
            (pl.col("_good_bad_norm") == "G") & (pl.col("_y_norm") == 0)
        ).sum().alias("n_G0"),
    ).row(0, named=True)
    return {name: int(value) for name, value in row.items()}


def score_single_step_lot_dataframe(
    data: pl.DataFrame | pl.LazyFrame,
    step_seq: Optional[Any] = None,
    params: Optional[ScoreParams] = None,
    max_wafers: int = 25,
) -> pl.DataFrame:
    """
    Score one step containing one or more root lots and return one Polars row.

    Required input columns are root_lot_id, wafer_id, good_bad, and y. If a
    step_seq column is present it must contain exactly one value; otherwise
    step_seq can be supplied as an argument and defaults to "input_step". All
    input lots are pooled into one count table and scored once for that step.
    max_wafers limits wafer_id within each root lot, not the number of lots.
    """
    params = params or ScoreParams()
    df = _to_polars_dataframe(data)
    root_lot_ids = _validate_single_step_lot_dataframe(
        df,
        max_wafers=max_wafers,
    )

    if "step_seq" in df.columns:
        if df.get_column("step_seq").null_count():
            raise ValueError(
                "step_seq must not contain missing values when provided as a column."
            )
        step_values = (
            df.get_column("step_seq").unique(maintain_order=True).to_list()
        )
        if len(step_values) != 1:
            raise ValueError(
                f"Expected exactly one step_seq value, got {step_values}."
            )
        inferred_step_seq = step_values[0]
        if step_seq is not None and step_seq != inferred_step_seq:
            raise ValueError(
                f"step_seq argument {step_seq!r} does not match "
                f"input {inferred_step_seq!r}."
            )
        step_seq = inferred_step_seq
    elif step_seq is None:
        step_seq = "input_step"

    result: Dict[str, Any] = score_process_counts(
        **_count_good_bad_context(df),
        params=params,
    )
    result["step_seq"] = step_seq
    result["root_lot_id"] = (
        root_lot_ids[0] if len(root_lot_ids) == 1 else None
    )
    result["root_lot_count"] = len(root_lot_ids)
    result["root_lot_ids"] = root_lot_ids
    result["input_rows"] = df.height
    return pl.DataFrame([result], infer_schema_length=None)


def score_process_dataframe(
    data: pl.DataFrame | pl.LazyFrame,
    params: Optional[ScoreParams] = None,
) -> pl.DataFrame:
    """Pool all lots within each step_seq and return one Polars row per step."""
    params = params or ScoreParams()
    df = _to_polars_dataframe(data)
    _validate_input_dataframe(df)
    if df.is_empty():
        return pl.DataFrame()

    normalized = _with_normalized_columns(df)
    counts_df = normalized.group_by("step_seq", maintain_order=True).agg(
        pl.col("root_lot_id").n_unique().alias("root_lot_count"),
        pl.col("root_lot_id")
        .cast(pl.String)
        .unique(maintain_order=True)
        .alias("root_lot_ids"),
        pl.len().alias("input_rows"),
        (
            (pl.col("_good_bad_norm") == "B") & (pl.col("_y_norm") == 1)
        ).sum().alias("n_B1"),
        (
            (pl.col("_good_bad_norm") == "B") & (pl.col("_y_norm") == 0)
        ).sum().alias("n_B0"),
        (
            (pl.col("_good_bad_norm") == "G") & (pl.col("_y_norm") == 1)
        ).sum().alias("n_G1"),
        (
            (pl.col("_good_bad_norm") == "G") & (pl.col("_y_norm") == 0)
        ).sum().alias("n_G0"),
    )

    rows: List[Dict[str, Any]] = []
    for count_row in counts_df.iter_rows(named=True):
        result: Dict[str, Any] = score_process_counts(
            n_B1=int(count_row["n_B1"]),
            n_B0=int(count_row["n_B0"]),
            n_G1=int(count_row["n_G1"]),
            n_G0=int(count_row["n_G0"]),
            params=params,
        )
        result["step_seq"] = count_row["step_seq"]
        result["root_lot_count"] = int(count_row["root_lot_count"])
        result["root_lot_ids"] = count_row["root_lot_ids"]
        result["input_rows"] = int(count_row["input_rows"])
        rows.append(result)

    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "score",
        descending=True,
    )


def make_case(
    step_seq: str,
    n_B1: int,
    n_B0: int,
    n_G1: int,
    n_G0: int,
    root_lot_id: Optional[str] = None,
) -> pl.DataFrame:
    """Create Polars wafer rows from count inputs for one process step."""
    _validate_counts(n_B1, n_B0, n_G1, n_G0)

    total = int(n_B1) + int(n_B0) + int(n_G1) + int(n_G0)
    root_lot_id = root_lot_id or "LOT01"
    good_bad = (
        ["bad"] * int(n_B1)
        + ["bad"] * int(n_B0)
        + ["good"] * int(n_G1)
        + ["good"] * int(n_G0)
    )
    y = (
        [1] * int(n_B1)
        + [0] * int(n_B0)
        + [1] * int(n_G1)
        + [0] * int(n_G0)
    )
    return pl.DataFrame(
        {
            "step_seq": [step_seq] * total,
            "root_lot_id": [root_lot_id] * total,
            "wafer_id": list(range(1, total + 1)),
            "good_bad": good_bad,
            "y": y,
        },
        schema={
            "step_seq": pl.String,
            "root_lot_id": pl.String,
            "wafer_id": pl.Int64,
            "good_bad": pl.String,
            "y": pl.Int64,
        },
    )


def _assert_close_to_zero(value: float, name: str, tol: float = 1e-12) -> None:
    assert abs(value) <= tol, f"{name} expected near zero, got {value}"


def _result_rows_by_step(result_df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["step_seq"]): row
        for row in result_df.iter_rows(named=True)
    }


def _run_sanity_checks(result_df: pl.DataFrame) -> None:
    by_id = _result_rows_by_step(result_df)

    assert by_id["lot_perfect_N25"]["score"] > by_id["medium_perfect_N10"]["score"]
    assert by_id["medium_perfect_N10"]["score"] > by_id["tiny_perfect_N2"]["score"]
    assert by_id["no_good_N5"]["score"] > by_id["tiny_single_bad_N1"]["score"]
    assert by_id["lot_perfect_N25"]["score"] > by_id["no_good_N5"]["score"]
    assert (
        by_id["lot_perfect_N25"]["score"]
        > by_id["noisy_good_in_y1_N25"]["score"]
    )
    assert (
        by_id["noisy_good_in_y1_N25"]["score"]
        > by_id["bad_missed_N25"]["score"]
    )
    assert by_id["all_y1_N25"]["score"] < by_id["no_good_N5"]["score"]
    assert by_id["no_good_N5"]["evidence_eff"] > 4.0 - 1e-2
    assert by_id["tiny_single_bad_N1"]["evidence_eff"] > 1.3

    _assert_close_to_zero(by_id["all_y0_N25"]["score"], "all_y0_N25 score")
    _assert_close_to_zero(by_id["no_bad_N25"]["score"], "no_bad_N25 score")


def plot_ranked_cases(
    df: pl.DataFrame | pl.LazyFrame,
    result_df: pl.DataFrame | pl.LazyFrame,
    output_path: Optional[str | Path] = None,
    close: bool = False,
    title: str = "Simulation Cases Ranked by Suspicion Score",
):
    """Plot Polars simulation rows in descending score order."""
    mpl_config_dir = (
        Path(tempfile.gettempdir()) / "wafer_process_suspicion_matplotlib"
    )
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if output_path is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _to_polars_dataframe(df)
    ranked = _to_polars_dataframe(result_df).sort("score", descending=True)
    n_cases = ranked.height
    fig_height = max(5.0, 1.75 * n_cases)
    fig, axes = plt.subplots(
        n_cases,
        1,
        figsize=(16, fig_height),
        constrained_layout=True,
    )
    if n_cases == 1:
        axes = [axes]

    state_style = {
        "G": {"color": "#2563eb", "marker": "o", "label": "Good wafer"},
        "B": {"color": "#dc2626", "marker": "X", "label": "Bad wafer"},
    }

    for rank, row in enumerate(ranked.iter_rows(named=True), start=1):
        ax = axes[rank - 1]
        step_seq = row["step_seq"]
        case_df = (
            df.filter(pl.col("step_seq") == step_seq)
            .select("root_lot_id", "wafer_id", "good_bad", "y")
            .sort("root_lot_id", "wafer_id", maintain_order=True)
            .with_row_index("x_pos", offset=1)
            .with_columns(
                pl.concat_str(
                    pl.col("root_lot_id").cast(pl.String),
                    pl.col("wafer_id").cast(pl.String),
                    separator="/",
                ).alias("wafer_key"),
                _normalize_good_bad_expr(),
            )
        )

        ax.axhspan(0.5, 1.35, color="#fee2e2", alpha=0.25, zorder=0)
        ax.axhspan(-0.35, 0.5, color="#dbeafe", alpha=0.18, zorder=0)
        ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=1)
        ax.axhline(1, color="#94a3b8", linewidth=0.8, zorder=1)

        for state, style in state_style.items():
            subset = case_df.filter(pl.col("_good_bad_norm") == state)
            if subset.is_empty():
                continue
            ax.scatter(
                subset.get_column("x_pos").to_numpy(),
                subset.get_column("y").cast(pl.Int64).to_numpy(),
                s=70,
                c=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.8,
                label=style["label"] if rank == 1 else None,
                zorder=3,
            )

        x_positions = case_df.get_column("x_pos").to_numpy()
        ax.set_xlim(0.4, case_df.height + 0.6)
        ax.set_ylim(-0.35, 1.35)
        ax.set_yticks([0, 1])
        ax.set_ylabel("y")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            case_df.get_column("wafer_key").to_list(),
            rotation=90,
            fontsize=6,
        )
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
        lot_end_positions = (
            case_df.group_by("root_lot_id", maintain_order=True)
            .agg(pl.col("x_pos").max())
            .get_column("x_pos")
            .to_numpy()
        )
        for lot_end in lot_end_positions[:-1]:
            ax.axvline(lot_end + 0.5, color="#64748b", linewidth=1.2, zorder=2)
        ax.set_title(
            f"{rank:02d}. step_seq={step_seq} | score={row['score']:.4g} | "
            f"quality={row['quality']:.3g} | evidence={row['evidence']:.3g}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

    axes[-1].set_xlabel("root_lot_id / wafer_id order")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
    if close:
        plt.close(fig)
    return fig


def make_simulation_dataframe() -> pl.DataFrame:
    """Build the requested wafer-level simulation cases as Polars."""
    cases = [
        make_case("lot_perfect_N25", n_B1=5, n_B0=0, n_G1=0, n_G0=20),
        make_case("tiny_perfect_N2", n_B1=1, n_B0=0, n_G1=0, n_G0=1),
        make_case("tiny_single_bad_N1", n_B1=1, n_B0=0, n_G1=0, n_G0=0),
        make_case("all_y1_N25", n_B1=5, n_B0=0, n_G1=20, n_G0=0),
        make_case("all_y0_N25", n_B1=0, n_B0=5, n_G1=0, n_G0=20),
        make_case("bad_missed_N25", n_B1=1, n_B0=4, n_G1=0, n_G0=20),
        make_case(
            "noisy_good_in_y1_N25",
            n_B1=5,
            n_B0=0,
            n_G1=8,
            n_G0=12,
        ),
        make_case("random_mix_N25", n_B1=2, n_B0=3, n_G1=4, n_G0=16),
        make_case("medium_perfect_N10", n_B1=2, n_B0=0, n_G1=0, n_G0=8),
        make_case("no_bad_N25", n_B1=0, n_B0=0, n_G1=0, n_G0=25),
        make_case("no_good_N5", n_B1=5, n_B0=0, n_G1=0, n_G0=0),
    ]
    return pl.concat(cases, how="vertical")


def make_multi_lot_simulation_dataframe() -> pl.DataFrame:
    """Build four Polars simulation cases, each pooled from three root lots."""
    case_specs = {
        "three_lot_perfect": {
            "LOT_A": (2, 0, 0, 6),
            "LOT_B": (2, 0, 0, 6),
            "LOT_C": (2, 0, 0, 6),
        },
        "three_lot_good_noise": {
            "LOT_A": (2, 0, 0, 6),
            "LOT_B": (2, 0, 3, 3),
            "LOT_C": (2, 0, 0, 6),
        },
        "three_lot_bad_missed": {
            "LOT_A": (2, 0, 0, 6),
            "LOT_B": (0, 2, 0, 6),
            "LOT_C": (2, 0, 0, 6),
        },
        "three_lot_random_mix": {
            "LOT_A": (1, 1, 1, 5),
            "LOT_B": (1, 1, 2, 4),
            "LOT_C": (1, 1, 1, 5),
        },
    }

    frames: List[pl.DataFrame] = []
    for step_seq, lot_specs in case_specs.items():
        for root_lot_id, counts in lot_specs.items():
            frames.append(
                make_case(
                    step_seq,
                    n_B1=counts[0],
                    n_B0=counts[1],
                    n_G1=counts[2],
                    n_G0=counts[3],
                    root_lot_id=root_lot_id,
                )
            )
    return pl.concat(frames, how="vertical")


def _run_multi_lot_sanity_checks(result_df: pl.DataFrame) -> None:
    """Verify that every case pools three lots and preserves expected ranking."""
    by_id = _result_rows_by_step(result_df)
    assert all(row["root_lot_count"] == 3 for row in by_id.values())
    assert all(row["input_rows"] == 24 for row in by_id.values())
    assert all(
        row["root_lot_ids"] == ["LOT_A", "LOT_B", "LOT_C"]
        for row in by_id.values()
    )

    assert (
        by_id["three_lot_perfect"]["score"]
        > by_id["three_lot_good_noise"]["score"]
    )
    assert (
        by_id["three_lot_perfect"]["score"]
        > by_id["three_lot_bad_missed"]["score"]
    )
    assert (
        by_id["three_lot_perfect"]["score"]
        > by_id["three_lot_random_mix"]["score"]
    )


def run_simulation_cases(
    params: Optional[ScoreParams] = None,
    plot_path: Optional[str | Path] = None,
) -> pl.DataFrame:
    """Build, print, and return Polars simulation results."""
    params = params or ScoreParams()
    df = make_simulation_dataframe()
    result_df = score_process_dataframe(df, params=params)
    _run_sanity_checks(result_df)

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        float_precision=6,
    ):
        print(result_df.select(DISPLAY_COLUMNS))
    print("\nSanity checks passed.")
    if plot_path is not None:
        plot_ranked_cases(
            df=df,
            result_df=result_df,
            output_path=plot_path,
            close=True,
        )
        print(f"Ranked wafer-context plot saved to: {plot_path}")
    return result_df


if __name__ == "__main__":
    run_simulation_cases()
