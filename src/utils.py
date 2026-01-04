import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -----------------------------
# Load data (CSV)
# -----------------------------
def load_yearly_csv(path, sep=";", decimal=".", thousands=None, source_col="Source", year_col="Year"):
    df = pd.read_csv(path, sep=sep, decimal=decimal, thousands=thousands)

    # convert everything to numeric except Source
    for c in df.columns:
        if c != source_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(year_col).reset_index(drop=True)


def load_categories_csv(path, sep=";", decimal=".", thousands=None):
    cats = pd.read_csv(path, sep=sep, decimal=decimal, thousands=thousands)

    # make sure key columns are numeric if present
    for c in ["Year", "Category", "Number of grants", "Percentage"]:
        if c in cats.columns:
            cats[c] = pd.to_numeric(cats[c], errors="coerce")

    return cats


# -----------------------------
# Metrics + summaries
# -----------------------------
def add_pot_metrics(
    yearly,
    apps_col="Applications received",
    award_col="Mean budget awarded grants (AUD)",
    pot_col="Total commitment ($AUD million)",
):
    out = yearly.copy()
    out["pot_needed_10pct_M"] = 0.10 * out[apps_col] * out[award_col] / 1e6
    out["gap_to_10pct_M"] = out["pot_needed_10pct_M"] - out[pot_col]
    return out


def category6_unfunded_summary(
    cats,
    yearly,
    funded_col="Grants funded",
    year_col="Year",
    category_col="Category",
    count_col="Number of grants",
    cat6=6,
    cat7=7,
):
    # counts per year/category
    pivot_cnt = cats.pivot_table(index=year_col, columns=category_col, values=count_col, aggfunc="sum").sort_index()
    idx = pivot_cnt.index

    cat6_s = pivot_cnt[cat6] if cat6 in pivot_cnt.columns else pd.Series(index=idx, data=np.nan, dtype=float)
    cat7_s = pivot_cnt[cat7] if cat7 in pivot_cnt.columns else pd.Series(index=idx, data=0.0, dtype=float)
    cat7_s = cat7_s.fillna(0)

    funded = yearly.set_index(year_col)[funded_col].reindex(idx).astype(float)

    summary = pd.DataFrame({
        year_col: idx,
        "Cat6": cat6_s.values,
        "Cat6plus7": (cat6_s.fillna(0) + cat7_s).values,
        "Funded_total": funded.values,
    })

    # lower bound: assume funded applications come from Cat6 first
    summary["Outstanding_unfunded_lowerbound"] = (summary["Cat6"] - summary["Funded_total"]).clip(lower=0)

    # percent (lower bound)
    summary["Outstanding_unfunded_pct_lowerbound"] = np.where(
        summary["Cat6"] > 0,
        summary["Outstanding_unfunded_lowerbound"] / summary["Cat6"] * 100,
        np.nan,
    )

    return summary


# -----------------------------
# Projection
# -----------------------------
def project_from_base(
    yearly,
    base_year=2025,
    horizon=5,
    pot_g=0.04,
    award_g=0.111,
    apps_g=-0.029,
    year_col="Year",
    pot_col="Total commitment ($AUD million)",
    award_col="Mean budget awarded grants (AUD)",
    apps_col="Applications received",
):
    base = yearly.loc[yearly[year_col] == base_year]
    if base.empty:
        raise ValueError(f"base_year={base_year} not found in yearly data")

    base = base.iloc[0]
    pot = float(base[pot_col])      # $M
    award = float(base[award_col])  # $
    apps = float(base[apps_col])    # count

    rows = []
    for i in range(horizon + 1):  # base_year..base_year+horizon
        yr = base_year + i
        funded_implied = pot * 1e6 / award
        success_implied = funded_implied / apps * 100
        pot_needed_10 = 0.10 * apps * award / 1e6

        rows.append({
            year_col: yr,
            "Pot_M": pot,
            "PotNeeded10pct_M": pot_needed_10,
            "Success_implied_pct": success_implied,
        })

        pot *= (1 + pot_g)
        award *= (1 + award_g)
        apps *= (1 + apps_g)

    return pd.DataFrame(rows)


# -----------------------------
# Plots
# -----------------------------
def plot_pot_vs_needed_with_projection(
    yearly,
    proj,
    start_year=2019,
    hist_end=2025,
    proj_end=2030,
    year_col="Year",
    pot_col="Total commitment ($AUD million)",
    needed_col="pot_needed_10pct_M",
    proj_pot_col="Pot_M",
    proj_needed_col="PotNeeded10pct_M",
    title="Actual vs required pot for 10% success (historical + projection)",
    figsize=(9.5, 5.2),
):
    h = yearly.copy()
    p = proj.copy()

    h[year_col] = pd.to_numeric(h[year_col], errors="coerce")
    p[year_col] = pd.to_numeric(p[year_col], errors="coerce")

    h = h[h[year_col].between(start_year, hist_end)].sort_values(year_col)
    p = p[p[year_col].between(hist_end, proj_end)].sort_values(year_col)

    fig, ax = plt.subplots(figsize=figsize)

    l1, = ax.plot(h[year_col].astype(int), h[pot_col], marker="o", label="Actual pot ($M)")
    ax.plot(p[year_col].astype(int), p[proj_pot_col], marker="o", linestyle="--",
            color=l1.get_color(), label="Projected pot ($M)")

    l2, = ax.plot(h[year_col].astype(int), h[needed_col], marker="o",
                  label="Pot needed for 10% success ($M)")
    ax.plot(p[year_col].astype(int), p[proj_needed_col], marker="o", linestyle="--",
            color=l2.get_color(), label="Projected pot needed for 10% ($M)")

    ax.set_xlabel("Year")
    ax.set_ylabel("$AUD million")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_cat6_unfunded(summary, mode="count", year_col="Year", figsize=(9.5, 5.2)):
    df = summary.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.sort_values(year_col)

    years = df[year_col].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "count":
        ax.bar(years, df["Outstanding_unfunded_lowerbound"])
        ax.set_ylabel("Count")
        ax.set_title("Category 6 ('Outstanding') unfunded applications (lower bound)")
    elif mode == "pct":
        ax.plot(years, df["Outstanding_unfunded_pct_lowerbound"], marker="o")
        ax.set_ylabel("% of Cat6 unfunded (lower bound)")
        ax.set_title("Category 6 ('Outstanding') unfunded applications (lower bound, %)")
    else:
        raise ValueError("mode must be 'count' or 'pct'")

    ax.set_xlabel("Year")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  
    fig.tight_layout()
    return fig, ax