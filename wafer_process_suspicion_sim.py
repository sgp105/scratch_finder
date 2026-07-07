from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom as scipy_hypergeom
except Exception:  # pragma: no cover - scipy is optional.
    scipy_hypergeom = None


@dataclass(frozen=True)
class ScoreParams:
    p_B: float = 0.95
    p_G: float = 0.20
    alpha_B: float = 0.7
    alpha_G: float = 0.3
    k_N: float = 8.0
    eps: float = 1e-12


DISPLAY_COLUMNS = [
    "process_id",
    "N",
    "n_B",
    "n_G",
    "m",
    "n_B1",
    "n_B0",
    "n_G1",
    "n_G0",
    "mean_likelihood",
    "quality",
    "p_tail",
    "evidence",
    "r_N",
    "score",
    "score_exact_alt",
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

    Degenerate / no-contrast cases return 1.0 so evidence becomes zero:
    no wafers, no Bad class, no Good class, no y=1 wafer, or all wafers y=1.
    """
    N = int(N)
    n_B = int(n_B)
    m = int(m)
    x_obs = int(x_obs)
    n_G = N - n_B

    if N <= 0 or n_B <= 0 or n_G <= 0 or m <= 0 or m >= N:
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


def score_process_counts(
    n_B1: int,
    n_B0: int,
    n_G1: int,
    n_G0: int,
    params: Optional[ScoreParams] = None,
) -> Dict[str, float]:
    """Score one process from its 2x2 state/context count table."""
    params = params or ScoreParams()
    _validate_counts(n_B1, n_B0, n_G1, n_G0)

    n_B1 = int(n_B1)
    n_B0 = int(n_B0)
    n_G1 = int(n_G1)
    n_G0 = int(n_G0)

    p_B = _safe_prob(params.p_B, params.eps)
    p_G = _safe_prob(params.p_G, params.eps)

    n_B = n_B1 + n_B0
    n_G = n_G1 + n_G0
    N = n_B + n_G
    m = n_B1 + n_G1

    w_B0 = math.log(p_B / (1.0 - p_B))
    w_G1 = math.log((1.0 - p_G) / p_G)

    if N == 0:
        log_likelihood = 0.0
        mean_likelihood = 0.0
        ideal_log_likelihood = 0.0
        r_N = 0.0
    else:
        log_likelihood = (
            n_B1 * math.log(p_B)
            + n_B0 * math.log(1.0 - p_B)
            + n_G1 * math.log(p_G)
            + n_G0 * math.log(1.0 - p_G)
        )
        mean_likelihood = math.exp(log_likelihood / N)
        ideal_log_likelihood = n_B * math.log(p_B) + n_G * math.log(1.0 - p_G)
        r_N = N / (N + params.k_N)

    penalty_raw = n_B0 * w_B0 + n_G1 * w_G1
    bad_miss_rate = n_B0 / n_B if n_B > 0 else 0.0
    good_false_context_rate = n_G1 / n_G if n_G > 0 else 0.0
    J_bal = (
        params.alpha_B * w_B0 * bad_miss_rate
        + params.alpha_G * w_G1 * good_false_context_rate
    )
    quality = math.exp(-J_bal)

    p_tail = hypergeom_tail_pvalue(N=N, n_B=n_B, m=m, x_obs=n_B1)
    evidence = 0.0 if p_tail >= 1.0 else -math.log(max(p_tail, params.eps))
    score = r_N * quality * evidence

    log_count_prob_alt = (
        log_comb(n_B, n_B1)
        + n_B1 * math.log(p_B)
        + n_B0 * math.log(1.0 - p_B)
        + log_comb(n_G, n_G1)
        + n_G1 * math.log(p_G)
        + n_G0 * math.log(1.0 - p_G)
    )
    count_prob_alt = float(math.exp(log_count_prob_alt)) if log_count_prob_alt > math.log(np.finfo(float).tiny) else 0.0
    score_exact_alt = r_N * quality * (-math.log(max(count_prob_alt, params.eps)))

    return {
        "N": N,
        "n_B": n_B,
        "n_G": n_G,
        "m": m,
        "n_B1": n_B1,
        "n_B0": n_B0,
        "n_G1": n_G1,
        "n_G0": n_G0,
        "p_B": p_B,
        "p_G": p_G,
        "w_B0": w_B0,
        "w_G1": w_G1,
        "log_likelihood": log_likelihood,
        "mean_likelihood": mean_likelihood,
        "ideal_log_likelihood": ideal_log_likelihood,
        "penalty_raw": penalty_raw,
        "J_bal": J_bal,
        "quality": quality,
        "p_tail": p_tail,
        "evidence": evidence,
        "r_N": r_N,
        "score": score,
        "log_count_prob_alt": log_count_prob_alt,
        "count_prob_alt": count_prob_alt,
        "score_exact_alt": score_exact_alt,
    }


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    required = {"process_id", "wafer_id", "state", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    states = set(df["state"].dropna().unique())
    invalid_states = states - {"G", "B"}
    if invalid_states:
        raise ValueError(f"Invalid state values: {sorted(invalid_states)}")

    y_values = set(df["y"].dropna().unique())
    invalid_y = y_values - {0, 1}
    if invalid_y:
        raise ValueError(f"Invalid y values: {sorted(invalid_y)}")


def score_process_dataframe(df: pd.DataFrame, params: Optional[ScoreParams] = None) -> pd.DataFrame:
    """Score all process_id groups in a wafer-level DataFrame."""
    params = params or ScoreParams()
    _validate_input_dataframe(df)

    rows: List[Dict[str, Any]] = []
    for process_id, group in df.groupby("process_id", sort=False):
        state = group["state"]
        y = group["y"].astype(int)
        result = score_process_counts(
            n_B1=int(((state == "B") & (y == 1)).sum()),
            n_B0=int(((state == "B") & (y == 0)).sum()),
            n_G1=int(((state == "G") & (y == 1)).sum()),
            n_G0=int(((state == "G") & (y == 0)).sum()),
            params=params,
        )
        result["process_id"] = process_id
        rows.append(result)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def make_case(process_id: str, n_B1: int, n_B0: int, n_G1: int, n_G0: int) -> pd.DataFrame:
    """Create wafer-level rows from count inputs for one process."""
    _validate_counts(n_B1, n_B0, n_G1, n_G0)

    rows: List[Dict[str, Any]] = []
    total = int(n_B1) + int(n_B0) + int(n_G1) + int(n_G0)
    wafer_id_width = max(2, len(str(total)))

    def add_rows(prefix: str, state: str, y: int, count: int) -> None:
        start = len(rows) + 1
        for offset in range(int(count)):
            rows.append(
                {
                    "process_id": process_id,
                    "wafer_id": f"W{start + offset:0{wafer_id_width}d}",
                    "state": state,
                    "y": y,
                }
            )

    add_rows("B1", "B", 1, n_B1)
    add_rows("B0", "B", 0, n_B0)
    add_rows("G1", "G", 1, n_G1)
    add_rows("G0", "G", 0, n_G0)
    return pd.DataFrame(rows, columns=["process_id", "wafer_id", "state", "y"])


def _assert_close_to_zero(value: float, name: str, tol: float = 1e-12) -> None:
    assert abs(value) <= tol, f"{name} expected near zero, got {value}"


def _run_sanity_checks(result_df: pd.DataFrame) -> None:
    by_id = result_df.set_index("process_id")
    score = by_id["score"]

    assert score["lot_perfect_N25"] > score["medium_perfect_N10"]
    assert score["medium_perfect_N10"] > score["tiny_perfect_N2"]
    assert score["lot_perfect_N25"] > score["noisy_good_in_y1_N25"]
    assert score["noisy_good_in_y1_N25"] > score["bad_missed_N25"]

    _assert_close_to_zero(score["all_y1_N25"], "all_y1_N25 score")
    _assert_close_to_zero(score["all_y0_N25"], "all_y0_N25 score")
    _assert_close_to_zero(score["no_bad_N25"], "no_bad_N25 score")
    _assert_close_to_zero(score["no_good_N5"], "no_good_N5 score")


def plot_ranked_cases(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    close: bool = False,
):
    """Plot each process in score order with wafer_id along x and context y on y."""
    mpl_config_dir = Path(tempfile.gettempdir()) / "wafer_process_suspicion_matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if output_path is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = result_df.sort_values("score", ascending=False).reset_index(drop=True)
    n_cases = len(ranked)
    fig_height = max(5.0, 1.75 * n_cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(16, fig_height), constrained_layout=True)
    if n_cases == 1:
        axes = [axes]

    state_style = {
        "G": {"color": "#2563eb", "marker": "o", "label": "Good wafer"},
        "B": {"color": "#dc2626", "marker": "X", "label": "Bad wafer"},
    }

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        ax = axes[rank - 1]
        process_id = row["process_id"]
        case_df = (
            df.loc[df["process_id"] == process_id, ["wafer_id", "state", "y"]]
            .sort_values("wafer_id", kind="stable")
            .reset_index(drop=True)
        )
        case_df["x_pos"] = np.arange(1, len(case_df) + 1)

        ax.axhspan(0.5, 1.35, color="#fee2e2", alpha=0.25, zorder=0)
        ax.axhspan(-0.35, 0.5, color="#dbeafe", alpha=0.18, zorder=0)
        ax.axhline(0, color="#94a3b8", linewidth=0.8, zorder=1)
        ax.axhline(1, color="#94a3b8", linewidth=0.8, zorder=1)

        for state, style in state_style.items():
            subset = case_df[case_df["state"] == state]
            if subset.empty:
                continue
            ax.scatter(
                subset["x_pos"],
                subset["y"].astype(int),
                s=70,
                c=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.8,
                label=style["label"] if rank == 1 else None,
                zorder=3,
            )

        ax.set_xlim(0.4, len(case_df) + 0.6)
        ax.set_ylim(-0.35, 1.35)
        ax.set_yticks([0, 1])
        ax.set_ylabel("y")
        ax.set_xticks(case_df["x_pos"])
        ax.set_xticklabels(case_df["wafer_id"], rotation=90, fontsize=7)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.5)
        ax.set_title(
            f"{rank:02d}. {process_id} | score={row['score']:.4g} | "
            f"quality={row['quality']:.3g} | evidence={row['evidence']:.3g}",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

    axes[-1].set_xlabel("wafer_id order")
    fig.suptitle("Simulation Cases Ranked by Suspicion Score", fontsize=14, fontweight="bold")
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


def make_simulation_dataframe() -> pd.DataFrame:
    """Build the requested wafer-level simulation cases."""
    cases = [
        make_case("lot_perfect_N25", n_B1=5, n_B0=0, n_G1=0, n_G0=20),
        make_case("tiny_perfect_N2", n_B1=1, n_B0=0, n_G1=0, n_G0=1),
        make_case("tiny_single_bad_N1", n_B1=1, n_B0=0, n_G1=0, n_G0=0),
        make_case("all_y1_N25", n_B1=5, n_B0=0, n_G1=20, n_G0=0),
        make_case("all_y0_N25", n_B1=0, n_B0=5, n_G1=0, n_G0=20),
        make_case("bad_missed_N25", n_B1=1, n_B0=4, n_G1=0, n_G0=20),
        make_case("noisy_good_in_y1_N25", n_B1=5, n_B0=0, n_G1=8, n_G0=12),
        make_case("random_mix_N25", n_B1=2, n_B0=3, n_G1=4, n_G0=16),
        make_case("medium_perfect_N10", n_B1=2, n_B0=0, n_G1=0, n_G0=8),
        make_case("no_bad_N25", n_B1=0, n_B0=0, n_G1=0, n_G0=25),
        make_case("no_good_N5", n_B1=5, n_B0=0, n_G1=0, n_G0=0),
    ]
    return pd.concat(cases, ignore_index=True)


def run_simulation_cases(
    params: Optional[ScoreParams] = None,
    plot_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build the requested simulation cases, print key columns, and return all results."""
    params = params or ScoreParams()
    df = make_simulation_dataframe()
    result_df = score_process_dataframe(df, params=params)
    _run_sanity_checks(result_df)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.float_format",
        "{:.6g}".format,
    ):
        print(result_df[DISPLAY_COLUMNS].to_string(index=False))
    print("\nSanity checks passed.")
    if plot_path is not None:
        plot_ranked_cases(df=df, result_df=result_df, output_path=plot_path, close=True)
        print(f"Ranked wafer-context plot saved to: {plot_path}")
    return result_df


if __name__ == "__main__":
    run_simulation_cases()
