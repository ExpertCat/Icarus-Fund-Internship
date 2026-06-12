import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt

def third_fridays(start="2018-01-01", end="2023-12-31"):
    months = pd.date_range(start=start, end=end, freq="MS") # A list of month start dates: 2018-01-01, 2018-02-01, 2018-03-01, ... (MS means month start)
    out = []
    for m in months:
        fridays = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")
        # pd.offsets.MonthEnd(0) means go to the month end of the current month. Example: if m = 2018-01-01, then m + MonthEnd(0) = 2018-01-31
        # freq="W-FRI" means weekly dates on Friday.
        # example: fridays = [2018-01-05, 2018-01-12, 2018-01-19, 2018-01-26]
        out.append(fridays[2]) #pick the third Friday out, 0, 1, 2 (third)
    return pd.DatetimeIndex(out)
    # In a DataFrame, the index is the thing on the far-left that identifies each row. A DatetimeIndex means those row labels are timestamps.

F = third_fridays("2018-01-01", "2023-12-31") # third Friday for Jan 2018 to Dec 2023
E_next = third_fridays("2018-02-01", "2024-01-31")  # next-month expiry aligned with F; third Friday for Feb 2018 to Jan 2024
#We want to pair up the formation date, the third Friday of this month to the expiration day of the third Friday next month

cal = pd.DataFrame({"F_t": F, "E_t1": E_next}) #create the table below
cal["month"] = cal["F_t"].dt.to_period("M").astype(str) #cal["F_t"].dt activates pandas “datetime accessor” methods
# .to_period("M") converts each timestamp into a monthly period (month granularity). e.g. 2018-01-19 → period 2018-01
#.astype(str) turns that period into a string "2018-01" This is useful for grouping/merging by month later.
cal.head()

# pandas: tabular data manipulation
# numpy: vectorized math (max, where, etc.)
# Path: OS-independent path handling for file locations

# ============================================================
# 0) Paths (edit only these)
# ============================================================
BASE_DIR   = Path(r"C:\Chris Academics\Icarus Fund Internship\Option Price Folder")
# BASE_DIR points to the root folder that contains daily OptionMetrics parquet files,
# organized by year subfolders (e.g., ...\2018\optionmetrics_2018-01-19_nonzeroOI.parquet)

MAP_PATH   = Path(r"C:\Chris Academics\Icarus Fund Internship\us_stock_permno_with_secid.parquet")
# MAP_PATH is the mapping file linking OptionMetrics secid -> CRSP permno
# with validity windows sdate..edate (because identifiers change over time)

STOCK_PATH = Path(r"C:\Chris Academics\Icarus Fund Internship\stock_price__1996_2022.parquet")
# STOCK_PATH is the underlying stock price file (CRSP-style), used to get S at expiration

# ============================================================
# 1) Load secid -> permno mapping (validity windows)
# ============================================================
link = pd.read_parquet(MAP_PATH).copy()
# Read the mapping parquet into a DataFrame and copy it so edits do not affect the original object

link["sdate"] = pd.to_datetime(link["sdate"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
# Convert sdate into a datetime column.
# - errors="coerce": invalid strings become NaT (missing datetime).
# - fillna(1900-01-01): if sdate is missing, treat it as "valid from very early"
#   so that the mapping can still be used (subject to the later validity filter).

link["edate"] = pd.to_datetime(link["edate"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
# Convert edate into a datetime column.
# - fillna(2099-12-31): if edate is missing, treat it as "valid until very late"
#   so the mapping can still be used (again subject to later validity filter).

link["secid"] = pd.to_numeric(link["secid"], errors="coerce").astype("Int64")
# Ensure secid is numeric (OptionMetrics identifier).
# - errors="coerce": non-numeric secid becomes NaN
# - Int64 (capital I): pandas nullable integer dtype, supports missing values

# ============================================================
# 2) OptionMetrics helpers
# ============================================================
def opt_path(d: pd.Timestamp) -> Path:
    # Given a date d (Timestamp), construct the exact parquet file path for that day.
    # Assumes the folder structure is BASE_DIR / {year} / optionmetrics_{YYYY-MM-DD}_nonzeroOI.parquet
    return BASE_DIR / f"{d.year}" / f"optionmetrics_{d.strftime('%Y-%m-%d')}_nonzeroOI.parquet"

def load_opt_day(d: pd.Timestamp) -> pd.DataFrame:
    # Load the option quote file for formation date d (one trading day).
    df = pd.read_parquet(opt_path(d))
    # Read the daily option quotes into a DataFrame.

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Ensure the quote date is a datetime column (some datasets store it as string/int).

    df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
    # Ensure option expiration date is a datetime column.

    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    # Compute the midquote price for each option:
    # mid = (bid + ask)/2, used as a proxy for “fair” execution price.

    df["spread"] = df["best_offer"] - df["best_bid"]
    # Compute quoted bid-ask spread for each option (liquidity proxy / trading friction).

    df["K"] = df["strike_price"] / 1000.0
    # OptionMetrics strike_price is scaled by 1000 (e.g., 45000 means $45.000).
    # Convert it back to dollar strike.

    return df
    # Return daily options DataFrame with extra columns mid, spread, K.

# ============================================================
# 3) Stock loader (CRSP-style prc can be negative)
# ============================================================
def load_stock(stock_path: Path) -> pd.DataFrame:
    # Load underlying stock prices and standardize key columns.
    stock = pd.read_parquet(stock_path).copy()
    # Read stock parquet and copy to avoid accidental mutation.

    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    # Ensure date is datetime.

    stock["permno"] = pd.to_numeric(stock["permno"], errors="coerce").astype("Int64")
    # Ensure permno is numeric and nullable-int (CRSP permanent identifier).

    stock["prc"] = pd.to_numeric(stock["prc"], errors="coerce")
    # Ensure price is numeric.

    stock["S"] = stock["prc"].abs()
    # CRSP convention: prc can be negative (sign can encode bid/ask or other flags).
    # Use abs(prc) as the price level S.

    return stock
    # Return DataFrame with standardized columns including S.

# ============================================================
# 4) Step 4 helper: attach S_{t'} at (or before) E_t1
# ============================================================
def attach_stock_price_at_expiry(selected: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    # Goal: for each selected straddle row (permno, E_t1),
    # attach the underlying stock price S observed at or before E_t1.

    out = selected.copy()
    # Work on a copy so we do not mutate the caller's DataFrame.

    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    # Ensure permno is correct dtype for merging.

    out["E_t1"] = pd.to_datetime(out["E_t1"], errors="coerce")
    # Ensure E_t1 is a datetime for merge_asof.

    stock_small = stock[["permno", "date", "S"]].dropna(subset=["permno", "date", "S"]).copy()
    # Keep only the needed columns from stock (permno, date, S).
    # Drop rows where any of these are missing.
    # Copy to avoid mutating original stock DataFrame.

    stock_small["permno"] = pd.to_numeric(stock_small["permno"], errors="coerce").astype("Int64")
    # Standardize permno.

    stock_small["date"] = pd.to_datetime(stock_small["date"], errors="coerce")
    # Standardize date.

    stock_small["S"] = pd.to_numeric(stock_small["S"], errors="coerce")
    # Standardize S numeric.

    stock_small = stock_small.rename(columns={"date": "stock_date"})
    # Rename date -> stock_date so that later we can keep both:
    # - out.E_t1 (target time)
    # - stock_small.stock_date (actual matched quote date)

    stock_small = stock_small.sort_values(["stock_date", "permno"]).reset_index(drop=True)
    # merge_asof requires that the merge "on" key be sorted.
    # Here, your on-key is stock_date (right_on) and E_t1 (left_on).
    # You also sort by permno as the group key used in by="permno".

    out = out.sort_values(["E_t1", "permno"]).reset_index(drop=True)
    # Sort the left table similarly by its on-key (E_t1) and by-key (permno).

    out = pd.merge_asof(
        out,
        stock_small,
        left_on="E_t1",
        # For each row in out, we want to match based on E_t1 time.

        right_on="stock_date",
        # Match against stock_small.stock_date.

        by="permno",
        # Enforce permno equality: only match stock prices from the same permno.

        direction="backward",
        # Use the latest stock_date <= E_t1 (the last observed price on/before expiry).

        allow_exact_matches=True,
        # If stock_date equals E_t1 exactly, allow that exact match.
    )
    # After this, out gains columns:
    # - stock_date: the actual matched stock observation date
    # - S: the matched stock price level

    return out
    # Return the augmented DataFrame.

# ============================================================
# 5) (Steps 1–3) Build selected straddles for one month
#     + (Step 4) intrinsic payoff using STOCK price
# ============================================================
def month_return_top20first(month: str, cal: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:
    # Compute one-month straddle return for a single "formation month" identified by `month`.
    # You:
    #  - choose top-20 liquid secids on formation date F_t
    #  - select one delta-neutral straddle per secid (best liquidity)
    #  - compute intrinsic value at E_t1 using stock price S at expiry
    #  - compute return R_t1 = (exit - entry) / entry

    F_t  = cal.loc[cal["month"] == month, "F_t"].iloc[0]
    # Look up formation date F_t for this month in the calendar table.

    E_t1 = cal.loc[cal["month"] == month, "E_t1"].iloc[0]
    # Look up next-month expiration date E_t1 for this month in the calendar table.

    opt = load_opt_day(F_t)
    # Load all option quotes on the formation date F_t.

    std = opt["expiry_indicator"].isna() | (opt["expiry_indicator"].astype(str).str.strip() == "")
    # Define "standard" options as those whose expiry_indicator is missing/blank.
    # This is a data-quality filter to avoid special expiries/flags.

    opt_F1 = opt[std & (opt["exdate"] == E_t1)].copy()
    # Keep only options that:
    # - are standard (std)
    # - expire on the target expiration date E_t1
    # This isolates the “next-month expiry” option chain.

    opt_F1 = opt_F1.dropna(
        subset=["best_bid","best_offer","mid","spread","delta","open_interest","K","secid","cp_flag","optionid"]
    )
    # Remove rows missing any critical pricing/greeks/IDs.
    # You cannot form a tradable straddle without bid/ask, mid, delta, OI, strike, ids, etc.

    opt_F1 = opt_F1[
        (opt_F1["best_offer"] >= opt_F1["best_bid"]) &
        (opt_F1["best_offer"] > 0) &
        (opt_F1["best_bid"] > 0) &
        (opt_F1["mid"] > 0)
    ].copy()
    # Additional quote sanity checks:
    # - ask must be >= bid (no crossed market)
    # - bid/ask must be positive
    # - mid must be positive

    opt_F1["secid"] = pd.to_numeric(opt_F1["secid"], errors="coerce").astype("Int64")
    # Standardize secid dtype.

    opt_F1 = opt_F1.dropna(subset=["secid"]).copy()
    # Drop any option quotes where secid could not be parsed.

    # ---------- Top-20 secids FIRST (must have valid permno on F_t) ----------
    liq = (
        opt_F1.groupby("secid")
        .agg(total_oi=("open_interest", "sum"),
             med_spread=("spread", "median"))
        .reset_index()
    )
    # Collapse option chain information into a per-secid liquidity summary:
    # - total_oi: sum open interest across all options (proxy for activity)
    # - med_spread: median bid-ask spread across options (proxy for transaction friction)

    liq = liq.dropna(subset=["med_spread"]).copy()
    # If med_spread missing, liquidity score is undefined; drop those secids.

    liq["liq_score"] = liq["total_oi"] / (1.0 + liq["med_spread"])
    # Liquidity score: higher OI and smaller spreads => higher score.
    # Add 1.0 in denominator to avoid dividing by 0 when spreads are tiny.

    cand = liq.merge(link[["secid","permno","sdate","edate"]], on="secid", how="left")
    # Attach permno mapping candidates for each secid.

    cand = cand[(F_t >= cand["sdate"]) & (F_t <= cand["edate"])].copy()
    # Enforce mapping validity:
    # only accept secid->permno links that are valid on the formation date F_t.

    cand = (
        cand.sort_values(["liq_score","sdate"], ascending=[False, False])
            .groupby("secid", as_index=False)
            .head(1)
    )
    # If a secid has multiple valid mapping windows, keep exactly one:
    # - sort by higher liquidity score first (usually same within secid)
    # - break ties by more recent sdate (prefer newest mapping)
    # - head(1) keeps the chosen mapping row per secid

    cand = cand.dropna(subset=["permno"]).copy()
    # Ensure mapping succeeded; drop secids without permno.

    top20 = cand.sort_values("liq_score", ascending=False).head(20).copy()
    # Select the 20 most liquid secids among those with valid permno mappings.

    top20_map = top20[["secid","permno"]].copy()
    # Keep a small mapping table for later merge.

    top20_secids = set(top20_map["secid"].tolist())
    # Make a Python set of the top20 secids for fast filtering.

    opt_F1 = opt_F1[opt_F1["secid"].isin(top20_secids)].copy()
    # Restrict the option quotes to only those belonging to the top-20 secids.

    # ---------- Build straddles only within top20 (Steps 2–3) ----------
    calls = opt_F1[
        (opt_F1["cp_flag"] == "C") &
        (opt_F1["open_interest"] > 0) &
        (opt_F1["delta"].between(0.25, 0.75, inclusive="both"))
    ].copy()
    # Keep call options:
    # - correct flag
    # - positive OI (tradable / non-empty market)
    # - delta filter (avoid extreme deep OTM/ITM calls)

    puts = opt_F1[
        (opt_F1["cp_flag"] == "P") &
        (opt_F1["open_interest"] > 0)
    ].copy()
    # Keep put options:
    # - correct flag
    # - positive OI
    # Note: you are NOT filtering put delta here; weights + later filters help.

    calls = calls[[
        "secid","symbol","root","date","exdate","K","strike_price","optionid","delta","mid","spread","open_interest"
    ]].rename(columns={
        "optionid":"call_optionid",
        "delta":"call_delta",
        "mid":"call_mid",
        "spread":"call_spread",
        "open_interest":"call_oi",
    })
    # Keep only essential call columns and rename them with call_* prefixes
    # to avoid column collisions when merging calls and puts.

    puts = puts[[
        "secid","exdate","K","strike_price","optionid","delta","mid","spread","open_interest"
    ]].rename(columns={
        "optionid":"put_optionid",
        "delta":"put_delta",
        "mid":"put_mid",
        "spread":"put_spread",
        "open_interest":"put_oi",
    })
    # Keep essential put columns and rename them with put_* prefixes.

    pairs = calls.merge(puts, on=["secid","exdate","K","strike_price"], how="inner")
    # Pair calls and puts into candidate straddles by matching:
    # - same underlying secid
    # - same expiration exdate (E_t1)
    # - same strike K (and raw strike_price)
    # This generates all possible call-put pairs at the same strike.

    pairs["combined_spread"] = pairs["call_spread"] + pairs["put_spread"]
    # Compute combined bid-ask spread for the straddle (liquidity proxy).

    pairs = pairs[(pairs["call_delta"] - pairs["put_delta"]).abs() > 1e-8].copy()
    # Avoid division by near-zero when computing delta-neutral weights.
    # If call_delta ≈ put_delta, the hedge weight formula becomes unstable.

    pairs["wC"] = (-pairs["put_delta"]) / (pairs["call_delta"] - pairs["put_delta"])
    # Delta-neutral weight on call in a two-leg combination:
    # choose wC such that wC*call_delta + wP*put_delta = 0 and wP=1-wC.

    pairs["wP"] = 1.0 - pairs["wC"]
    # The put weight is the remainder so weights sum to 1.

    pairs = pairs[
        pairs["wC"].between(0, 1, inclusive="both") &
        pairs["wP"].between(0, 1, inclusive="both")
    ].copy()
    # Keep only economically sensible convex-combination weights:
    # both weights between 0 and 1 (no leverage / no negative weights).

    pairs["entry_mid"] = pairs["wC"] * pairs["call_mid"] + pairs["wP"] * pairs["put_mid"]
    # Entry premium of the delta-neutral straddle (weighted midquotes).

    best = (
        pairs.sort_values(["secid","combined_spread","entry_mid"], ascending=[True, True, True])
             .groupby("secid", as_index=False)
             .head(1)
             .copy()
    )
    # Choose ONE straddle per secid:
    # - primary: smallest combined spread (most liquid)
    # - tie-break: smallest entry_mid (cheaper premium, often less extreme strike)
    # - head(1): keep the best row per secid

    best["month"] = month
    # Attach formation month label.

    best["F_t"] = F_t
    # Attach formation date.

    best["E_t1"] = E_t1
    # Attach target expiration date.

    selected = best[[
        "month","F_t","E_t1",
        "secid","symbol","root",
        "K","strike_price",
        "call_optionid","put_optionid",
        "call_delta","put_delta","wC","wP",
        "call_mid","put_mid","entry_mid",
        "call_spread","put_spread","combined_spread",
        "call_oi","put_oi"
    ]].copy()
    # Build the final selected straddle table for this month,
    # keeping all variables needed for later steps.

    selected = selected.merge(top20_map, on="secid", how="left")
    # Attach permno to each selected straddle (secid->permno mapping from top20 list).

    # =======================================================
    # Step 4 (correct): use STOCK price at E_t1 to compute intrinsic payoff
    # =======================================================
    selected = attach_stock_price_at_expiry(selected, stock)
    # Add stock_date and underlying price S for each row using merge_asof at/before E_t1.

    selected["S"] = pd.to_numeric(selected["S"], errors="coerce")
    # Ensure S is numeric.

    selected["K"] = pd.to_numeric(selected["K"], errors="coerce")
    # Ensure strike is numeric.

    selected["entry_mid"] = pd.to_numeric(selected["entry_mid"], errors="coerce")
    # Ensure entry premium is numeric.

    selected["payoff_c"] = np.maximum(selected["S"] - selected["K"], 0.0)
    # Call intrinsic payoff at expiry: max(S - K, 0).

    selected["payoff_p"] = np.maximum(selected["K"] - selected["S"], 0.0)
    # Put intrinsic payoff at expiry: max(K - S, 0).

    selected["exit_intrinsic"] = selected["wC"] * selected["payoff_c"] + selected["wP"] * selected["payoff_p"]
    # Exit value of your delta-neutral straddle at expiry, using the same weights.

    valid = (
        selected["entry_mid"].notna() & (selected["entry_mid"] > 0) &
        selected["S"].notna() & (selected["S"] > 0) &
        selected["K"].notna() & (selected["K"] > 0)
    )
    # Define rows where return computation is meaningful:
    # - entry premium positive
    # - stock price positive
    # - strike positive

    selected["R_t1"] = np.where(
        valid,
        (selected["exit_intrinsic"] - selected["entry_mid"]) / selected["entry_mid"],
        np.nan
    )
    # Compute straddle return from formation to expiration:
    # R = (exit - entry) / entry.
    # If invalid, store NaN.

    return selected[[
        "month","F_t","E_t1","secid","permno","symbol","root",
        "K","entry_mid","stock_date","S",
        "payoff_c","payoff_p","exit_intrinsic","R_t1"
    ]].copy()
    # Return the monthly selected straddles + realized return and key intermediate fields.

# ============================================================
# 6) Run across months
#    IMPORTANT: this assumes you already have your calendar table
#    in a variable named `cal` with columns: ["F_t","E_t1","month"]
# ============================================================
stock = load_stock(STOCK_PATH)
# Load the entire stock price panel once (efficient) and reuse it for all months.

results = []
# Create a list to store each month’s selected-straddle DataFrame.

for m in cal["month"]:
    # Loop over each month label in your calendar table.
    try:
        dfm = month_return_top20first(m, cal, stock)
        # Compute the selected straddles and R_t1 for month m.

        results.append(dfm)
        # Store the result so we can concatenate later.

        print(m, "ok", "n=", len(dfm), "missing=", int(dfm["R_t1"].isna().sum()))
        # Quick progress report:
        # - n = how many straddles were produced (aim ~20)
        # - missing = how many have NaN returns (data issues / missing prices)

    except FileNotFoundError as e:
        print(m, "SKIP missing option file:", e)
        # If the daily OptionMetrics parquet for the formation date is missing,
        # skip the month rather than crashing the whole run.

all_monthly = pd.concat(results, ignore_index=True)
# Stack all monthly DataFrames into one big panel dataset across months.

print(all_monthly.head())
# Show the first few rows for a sanity check.

print("shape:", all_monthly.shape)
# Print (rows, columns) to confirm how much data was produced.

# ============================================================
# Step 5: MOM_{r,t} = avg of past 11 monthly returns excluding t-1
# i.e. mean(R_{t-12}, ..., R_{t-2})
#
# Interpretation:
# - For each stock r (identified by permno) and each evaluation month t,
#   compute a momentum signal MOM_{r,t} using ONLY "past" realized straddle returns.
# - Specifically, we use 11 months of returns, but we SKIP the most recent month (t-1).
#   That is why we use shift(2): it pushes the series so that at time t we only see up to t-2.
#
# Implementation plan:
# 1) Ensure month is sortable in true calendar order (not lexicographic string order).
# 2) Sort within each permno by month so rolling windows are correct.
# 3) Within each permno:
#       MOM_{r,t} = mean( R_{r,t-12}, ..., R_{r,t-2} )
#    achieved by:
#       - shift(2): aligns time t with value from t-2 (and earlier)
#       - rolling(window=11): takes 11 values (t-12 ... t-2)
#       - min_periods=8: require at least 8 valid months to compute momentum
# ============================================================

# Work off your existing all_monthly (does NOT overwrite it)
step5 = all_monthly.copy()             # Make a copy so you don't mutate all_monthly later by accident

# Ensure consistent types
step5["permno"] = (
    pd.to_numeric(step5["permno"], errors="coerce")  # Convert permno to numeric; invalid strings -> NaN
      .astype("Int64")                               # Use pandas nullable integer (keeps NaN as <NA>)
)

step5["secid"]  = (
    pd.to_numeric(step5["secid"], errors="coerce")   # Convert secid to numeric; invalid -> NaN
      .astype("Int64")                               # Nullable Int64 for consistency
)

step5["R_t1"]   = (
    pd.to_numeric(step5["R_t1"], errors="coerce")    # Convert return to float; invalid -> NaN
)
# Note: R_t1 should already be float-like, but this prevents silent string issues.

# Use a real sortable month index internally
step5["_month_ts"] = (
    pd.PeriodIndex(step5["month"].astype(str), freq="M")  # Interpret "month" as monthly period (YYYY-MM)
      .to_timestamp()                                     # Convert monthly periods to timestamp (typically first day of month)
)
# Why this matters:
# - If "month" stays as string, sorting may be wrong (e.g., "2019-10" < "2019-2" lexicographically).
# - PeriodIndex ensures correct chronological ordering.

# Sort before rolling
step5 = (
    step5.sort_values(["permno", "_month_ts"])  # Sort by stock first, then by time
         .reset_index(drop=True)                # Reset row index after sort (cleaner downstream)
)
# Why sorting matters:
# - rolling() uses the *row order*.
# - If months are not properly sorted, your momentum windows will be garbage.

# MOM_{r,t} = mean of (t-12 ... t-2), so: shift(2) then rolling window=11
step5["MOM_rt"] = (
    step5.groupby("permno", group_keys=False)["R_t1"]      # For each permno, take its time series of R_t1
         .apply(                                           # Apply a function to each permno's series
             lambda s: (
                 s.shift(2)                                # Shift by 2: at month t, value becomes R_{t-2}
                  .rolling(window=11, min_periods=8)       # Take last 11 shifted values: (t-12 ... t-2)
                  .mean()                                  # Average them -> MOM_{t}
             )
         )
)

# Detailed timing check (important):
# Let s_t = R_t1 at month t (your monthly straddle return measured over t -> t+1).
# After shift(2), at row for month t you see s_{t-2}.
# A rolling window of 11 months on shifted series at month t averages:
#   s_{t-2}, s_{t-3}, ..., s_{t-12}  (11 values)
# which is exactly mean(R_{t-12} ... R_{t-2}).
#
# min_periods=8 means:
# - If fewer than 8 non-NaN values exist in that 11-month window, MOM_rt becomes NaN.
# - This is your "need at least 8 valid months" rule.

# Deliverable 1: table
mom_table = step5[["permno", "secid", "month", "MOM_rt"]].copy()
# This creates a compact table containing only:
# - permno: stock identifier
# - secid: option identifier (may map to permno via your link table)
# - month: evaluation month t
# - MOM_rt: the momentum signal computed at month t

# Deliverable 2: summary stats
print("=== MOM summary stats ===")          # Label so output is readable in notebook logs
print(mom_table["MOM_rt"].describe())      # Count/mean/std/min/quantiles/max (ignores NaNs)
print(mom_table.head(5))                   # Show first 5 rows to sanity check structure

# Deliverable 3: heatmap (keep it readable by limiting permnos)
MAX_ROWS = 60                               # Cap number of permnos shown to keep the heatmap readable

mom_nonnull = (
    mom_table.dropna(subset=["MOM_rt"]).copy()  # Keep only rows where MOM_rt exists (remove NaNs)
)
# Why drop NaNs:
# - Early months won't have enough history (shift+rolling), so MOM_rt is NaN.
# - Heatmap with tons of NaNs looks empty/ugly and adds noise.

mom_nonnull["_month_ts"] = (
    pd.PeriodIndex(mom_nonnull["month"].astype(str), freq="M").to_timestamp()
)
# We rebuild a timestamp column for plotting columns in the pivot table.

print(mom_nonnull.head(5))                  # Quick sanity check: do we have MOM_rt and month_ts as expected?

top_permnos = (
    mom_nonnull["permno"]
    .value_counts()                         # Count how many non-null MOM observations per permno
    .head(MAX_ROWS)                         # Keep the top 60 permnos with the most MOM observations
    .index                                 # Extract the permno values
)
# Why choose "most observations":
# - Some permnos may have sparse data (missing months).
# - For a clean heatmap, you want permnos that exist across many months.

heat = (
    mom_nonnull[mom_nonnull["permno"].isin(top_permnos)]     # Filter to only those permnos
    .pivot_table(                                            # Create matrix: rows=permno, cols=month, values=MOM_rt
        index="permno",
        columns="_month_ts",
        values="MOM_rt",
        aggfunc="mean"                                       # If duplicates exist per permno-month, average them
    )
    .sort_index()                                            # Sort permnos (row order) ascending for stable display
)
# Interpretation of heat:
# - Each row = one permno
# - Each column = a month timestamp
# - Each cell = MOM_rt value at that permno-month

plt.figure()                                                 # New figure so it doesn't reuse old plot
plt.imshow(
    heat.values,                                             # Use the underlying 2D numeric array for image plotting
    aspect="auto",                                           # Stretch to fit figure (so you can see all months)
    interpolation="nearest"                                  # Do not smooth values; show raw cells
)
plt.title("MOM_rt heatmap (permno x month)")                 # Title explains what rows/cols represent
plt.xlabel("Month")                                          # X-axis label (columns)
plt.ylabel("permno")                                         # Y-axis label (rows)
plt.colorbar()                                               # Add legend mapping color -> MOM value
plt.show()                                                   # Render plot

# Clean internal helper col (keep step5 tidy for next steps)
step5 = step5.drop(columns=["_month_ts"])
# Remove temporary timestamp column:
# - You already have "month" in original format.
# - This avoids carrying extra columns into later steps (Step 6/7).

# =========================  # section header: Step 6 block
# Step 6 (REVISED): Top-3 / Bottom-3 selection with fallback  # what this block does
# Correct timing: sort on MOM_{t}, evaluate R_{t+1}  # timing convention (no look-ahead)
# Requires: step5 with columns ["month","permno","R_t1","MOM_rt"]  # required inputs
# Produces: LS_spread, LS_5050, membership (for Step 7 and reporting)  # main outputs
# =========================  # section header end

N_LONG  = 3  # number of longs to select each month
N_SHORT = 3  # number of shorts to select each month

panel6 = step5[["month", "permno", "R_t1", "MOM_rt"]].copy()  # keep only columns needed for Step 6 and copy

# Standardize types  # ensure consistent dtypes for sorting/rolling/groupby
panel6["permno"] = pd.to_numeric(panel6["permno"], errors="coerce").astype("Int64")  # force permno to nullable integer
panel6["R_t1"]   = pd.to_numeric(panel6["R_t1"], errors="coerce")  # force monthly return to numeric (NaN if bad)
panel6["MOM_rt"] = pd.to_numeric(panel6["MOM_rt"], errors="coerce")  # force momentum signal to numeric
panel6["month"]  = pd.PeriodIndex(panel6["month"].astype(str), freq="M")  # convert month strings to monthly PeriodIndex

# 1) Build next-month return R_{t+1} within each permno  # align returns so ranking at t maps to realized return at t+1
panel6 = panel6.sort_values(["permno", "month"]).reset_index(drop=True)  # sort so shift(-1) is correct within permno
panel6["R_next"] = panel6.groupby("permno")["R_t1"].shift(-1)  # realized next-month return for each permno

# 2) Optional stabilization / winsorization for portfolio mechanics  # cap extreme returns for robustness
# (Keep your original caps if you like)  # note: you can change CAP_LO/CAP_HI if desired
CAP_LO, CAP_HI = -0.95, 5.0  # lower/upper cap applied to next-month returns
panel6["R_next_cap"] = panel6["R_next"].clip(lower=CAP_LO, upper=CAP_HI)  # capped next-month return used in portfolio math

# 3) We need MOM_t to rank. Keep MOM non-null.  # drop rows that cannot be ranked by momentum
panel6 = panel6.dropna(subset=["permno", "month", "MOM_rt"]).copy()  # keep only rows with valid ranking signal

def pick_top_bottom_with_fallback(df_m: pd.DataFrame, n_long: int, n_short: int):  # helper: pick Top/Bottom with valid returns
    """
    For a single month t:
      - rank by MOM_rt (desc for long, asc for short)
      - select first n_long names with valid R_next_cap
      - select first n_short names from the bottom with valid R_next_cap
    """
    # Sort for ranking  # order by MOM for long selection
    df_sorted = df_m.sort_values(["MOM_rt", "permno"], ascending=[False, True]).copy()  # high MOM first; tie-break by permno

    # Long: go down the ranked list, keep only those with valid next-month return  # ensure portfolio uses realized R_next_cap
    longs = df_sorted[df_sorted["R_next_cap"].notna()].head(n_long).copy()  # take top n_long with non-missing next-month return

    # Short: go up from the bottom of the ranked list  # low MOM names are short candidates
    df_sorted_asc = df_m.sort_values(["MOM_rt", "permno"], ascending=[True, True]).copy()  # low MOM first; tie-break by permno
    shorts = df_sorted_asc[df_sorted_asc["R_next_cap"].notna()].head(n_short).copy()  # take bottom n_short with non-missing next-month return

    # If overlap occurs (possible in tiny universes), drop overlaps from shorts then refill  # avoid being both long and short
    if len(longs) > 0 and len(shorts) > 0:  # only check overlap if both sides non-empty
        overlap = set(longs["permno"].tolist()) & set(shorts["permno"].tolist())  # identify permnos selected on both sides
        if overlap:  # if any overlap exists
            shorts = shorts[~shorts["permno"].isin(overlap)].copy()  # remove overlapping names from shorts
            # refill shorts if needed  # backfill shorts from next-best remaining short candidates
            refill = (  # build refill candidates for shorts
                df_sorted_asc[df_sorted_asc["R_next_cap"].notna() & ~df_sorted_asc["permno"].isin(set(longs["permno"])) ]  # eligible shorts not in longs
                .head(n_short)  # take first n_short eligible (acts as refill list)
                .copy()  # materialize copy
            )  # end refill construction
            shorts = refill  # replace shorts with refilled selection

    long_ret  = float(longs["R_next_cap"].mean())  if len(longs)  > 0 else np.nan  # average capped next-month return of long leg
    short_ret = float(shorts["R_next_cap"].mean()) if len(shorts) > 0 else np.nan  # average capped next-month return of short leg

    # membership rows  # tag rows so you can audit membership later
    mem_long = longs.assign(side="LONG")  # label long selections
    mem_short = shorts.assign(side="SHORT")  # label short selections

    # return both  # output (long_ret, short_ret, membership_table)
    return long_ret, short_ret, pd.concat([mem_long, mem_short], ignore_index=True)  # combine long+short membership rows

# 4) Apply month by month  # loop over each formation month t
rows = []  # stores one row per month with portfolio returns and counts
members = []  # stores membership rows across all months
for m, df_m in panel6.groupby("month", sort=True):  # iterate month-by-month in chronological order
    long_ret, short_ret, mem = pick_top_bottom_with_fallback(df_m, N_LONG, N_SHORT)  # select long/short and compute their mean returns

    rows.append({  # store month-level outputs
        "month": m,  # formation month t
        "n_long": int(mem[mem["side"]=="LONG"].shape[0]),  # realized count of longs (can be < N_LONG if missing data)
        "n_short": int(mem[mem["side"]=="SHORT"].shape[0]),  # realized count of shorts (can be < N_SHORT if missing data)
        "long_ret_next": long_ret,  # average capped next-month return for the long leg
        "short_ret_next": short_ret,  # average capped next-month return for the short leg
        "LS_spread": (long_ret - short_ret) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # winner-minus-loser spread
        "LS_5050": (0.5*(long_ret - short_ret)) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # self-financing 50/50 long-short return
    })  # end month-level dict

    mem = mem[["month","permno","MOM_rt","R_next","R_next_cap","side"]].copy()  # keep only membership columns you care about
    members.append(mem)  # accumulate membership rows

step6_summary = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)  # assemble month-level summary table and sort by month
membership = pd.concat(members, ignore_index=True) if len(members) else pd.DataFrame()  # assemble membership table (or empty if none)

# 5) Output series for Step 7 (match your existing variable names)  # create time series objects for evaluation/plots
LS_spread = step6_summary.set_index("month")["LS_spread"]  # spread series indexed by month
LS_5050   = step6_summary.set_index("month")["LS_5050"]  # 50/50 long-short return series indexed by month

print("=== Step 6 (Top/Bottom 3) Summary ===")  # header print
print("Months with LS_5050:", int(LS_5050.notna().sum()))  # how many months produced a valid 50/50 return
print("Avg n_long / n_short:",  # print average realized selection sizes (may differ from 3/3)
      float(step6_summary["n_long"].mean()),  # average number of longs used
      float(step6_summary["n_short"].mean()))  # average number of shorts used

print("\nLS_5050 mean/std:",  # print mean and std of LS_5050
      float(LS_5050.dropna().mean()),  # mean monthly long-short return
      float(LS_5050.dropna().std(ddof=1)))  # std dev of monthly long-short return

# 6) Plots (same as before)  # visualize cumulative and distribution
cum_5050_add = LS_5050.dropna().cumsum()  # additive cumulative sum of monthly LS_5050 returns

plt.figure()  # start a new figure for cumulative plot
plt.plot(cum_5050_add.index.to_timestamp(), cum_5050_add.values)  # plot cumulative additive returns over time
plt.title("Cumulative 50/50 Long-Short (additive) — Top3/Bottom3")  # title for cumulative plot
plt.xlabel("Month")  # x-axis label
plt.ylabel("Cumulative sum")  # y-axis label
plt.show()  # render the cumulative plot

plt.figure()  # start a new figure for histogram
plt.hist(pd.to_numeric(LS_5050, errors="coerce").dropna().values, bins=30)  # histogram of monthly LS_5050 returns
plt.title("Histogram — LS_5050 (Top3/Bottom3)")  # title for histogram
plt.xlabel("Monthly return")  # x-axis label
plt.ylabel("Count")  # y-axis label
plt.show()  # render the histogram

# Deliverables you may want to inspect  # quick sanity-check views
display(step6_summary.head(10))  # show first 10 months of summary
display(membership.head(20))  # show first 20 membership rows

# Convenience subsets like before  # split membership by side for later analysis
QH = membership[membership["side"] == "LONG"].copy()  # long-side membership table
QL = membership[membership["side"] == "SHORT"].copy()  # short-side membership table


# =========================                                    # section header separator (comment only)
# Step 7: Evaluate Strategy Performance (continuation of your Step 6)  # describes what this step does
# Uses your Step 6 outputs:                                     # indicates required inputs
#   - LS_spread (winner - loser spread on next-month returns)   # spread long-short series from Step 6
#   - LS_5050   (50/50 self-financing long-short return on next-month returns)  # normalized long-short series
# =========================                                    # end of section header

def eval_series(R: pd.Series, name: str) -> dict:              # define a helper that computes performance stats for a return Series
    R = R.dropna().copy()                                      # drop missing values and copy to avoid mutating original
    R = pd.to_numeric(R, errors="coerce").dropna()             # force numeric values; non-numeric -> NaN; then drop NaNs

    T = int(R.shape[0])                                        # number of observations (months)
    mean_m = float(R.mean())                                   # average monthly return
    vol_m  = float(R.std(ddof=1))                               # monthly sample volatility (std dev with ddof=1)
    sharpe_m = mean_m / vol_m if vol_m > 0 else np.nan          # monthly Sharpe ratio (rf assumed 0); NaN if vol=0
    tstat = mean_m / (vol_m / np.sqrt(T)) if (vol_m > 0 and T > 1) else np.nan  # t-stat of mean return; requires vol>0 and T>1

    ann_ret = 12.0 * mean_m                                    # annualized return (linear scaling by 12 months)
    ann_vol = np.sqrt(12.0) * vol_m                             # annualized volatility (sqrt-time scaling)
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan   # annualized Sharpe ratio; NaN if ann_vol=0

    # Can we compound? only if (1+R) > 0 for all months         # comment: geometric compounding requires positive gross returns each month
    can_compound = bool(((1.0 + R) > 0).all())                  # True if every month has (1+R)>0, else False

    out = {                                                     # start a dict to store the computed statistics
        "name": name,                                           # store label/name for this series
        "T": T,                                                 # store number of months
        "mean_monthly": mean_m,                                 # store mean monthly return
        "vol_monthly": vol_m,                                   # store monthly volatility
        "sharpe_monthly": sharpe_m,                              # store monthly Sharpe
        "tstat": tstat,                                         # store t-stat of the mean
        "ann_ret": ann_ret,                                     # store annualized return
        "ann_vol": ann_vol,                                     # store annualized volatility
        "ann_sharpe": ann_sharpe,                               # store annualized Sharpe
        "min": float(R.min()) if T > 0 else np.nan,              # store worst month (min); NaN if no data
        "max": float(R.max()) if T > 0 else np.nan,              # store best month (max); NaN if no data
        "can_compound": can_compound,                           # store whether compounding is valid
    }                                                           # end dict literal
    return out                                                  # return the dict of summary statistics

def plot_cumulative(R: pd.Series, title: str):                  # define a helper to plot cumulative performance for a return Series
    R = R.dropna().copy()                                       # drop missing values and copy for safety
    if isinstance(R.index, pd.PeriodIndex):                     # if the index is monthly PeriodIndex (common in your pipeline)
        x = R.index.to_timestamp()                              # convert PeriodIndex to timestamps for plotting on x-axis
    else:                                                       # otherwise (already datetime-like)
        x = pd.to_datetime(R.index)                             # try to coerce index into datetimes for plotting

    # If any (1+R)<=0, geometric compounding breaks. Plot BOTH:  # comment: compounded wealth curve is invalid if any month return <= -100%
    can_compound = bool(((1.0 + R) > 0).all())                  # check if compounding is mathematically defined

    plt.figure()                                                # create a new figure for additive cumulative plot
    plt.plot(x, R.cumsum().values)                              # plot cumulative additive sum of returns over time
    plt.title(title + " — Cumulative (Additive Sum)")           # set title for additive cumulative plot
    plt.xlabel("Month")                                         # label x-axis
    plt.ylabel("Cumulative sum")                                # label y-axis
    plt.show()                                                  # render the additive plot

    if can_compound:                                            # if compounding is valid
        cumprod = (1.0 + R).cumprod() - 1.0                     # compute compounded cumulative return series
        plt.figure()                                            # create a new figure for compounded plot
        plt.plot(x, cumprod.values)                             # plot compounded cumulative return over time
        plt.title(title + " — Cumulative (Compounded)")         # set title for compounded cumulative plot
        plt.xlabel("Month")                                     # label x-axis
        plt.ylabel("Cumulative return")                         # label y-axis
        plt.show()                                              # render the compounded plot
    else:                                                       # if compounding is invalid
        print(f"WARNING: Cannot compound {title} because some months have (1+R) <= 0.")  # print warning explaining why no compounded plot

def plot_hist(R: pd.Series, title: str, bins: int = 30):        # define a helper to plot histogram of returns
    R = pd.to_numeric(R, errors="coerce").dropna()              # coerce to numeric, drop non-numeric and NaNs
    plt.figure()                                                # create a new figure for histogram
    plt.hist(R.values, bins=bins)                               # plot histogram of monthly return values
    plt.title("Histogram — " + title)                           # set histogram title
    plt.xlabel("Monthly return")                                # label x-axis
    plt.ylabel("Count")                                         # label y-axis
    plt.show()                                                  # render the histogram plot

# ------------------------------------------------------------  # separator for readability
# 1) Choose which LS series to evaluate                          # describes section purpose
#    - LS_5050 is the "portfolio return" (self-financing 50/50)   # clarifies interpretation of LS_5050
#    - LS_spread is a spread (not a normalized portfolio return)  # clarifies interpretation of LS_spread
# ------------------------------------------------------------  # end separator
if "LS_5050" not in globals() or "LS_spread" not in globals():  # check that Step 6 created required series in the global namespace
    raise NameError("Step 7 expects LS_5050 and LS_spread from your Step 6 cell.")  # stop execution with clear error message if missing

# ------------------------------------------------------------  # separator for readability
# 2) Evaluate both (recommended)                                  # describes section purpose
# ------------------------------------------------------------  # end separator
stats_5050   = eval_series(LS_5050,   "LS_5050 (50/50 self-financing)")  # compute stats dict for LS_5050 series
stats_spread = eval_series(LS_spread, "LS_spread (winners - losers spread)")  # compute stats dict for LS_spread series

summary = pd.DataFrame([stats_5050, stats_spread])              # convert both stats dicts into a 2-row DataFrame

print("\n=== Step 7 Summary Table ===")                         # print a header label before displaying the summary table
display(summary)                                                # display the summary DataFrame (works in notebooks/IPython; may need print() in plain scripts)

# ------------------------------------------------------------  # separator for readability
# 3) Plot cumulative performance + histograms                     # describes section purpose
# ------------------------------------------------------------  # end separator
plot_cumulative(LS_5050, "LS_5050 (50/50 self-financing)")      # plot additive and (if valid) compounded cumulative for LS_5050
plot_hist(LS_5050, "LS_5050 (50/50 self-financing)", bins=30)   # plot histogram of monthly LS_5050 returns

plot_cumulative(LS_spread, "LS_spread (spread, not normalized)")# plot additive and (if valid) compounded cumulative for LS_spread
plot_hist(LS_spread, "LS_spread (spread, not normalized)", bins=30)  # plot histogram of monthly LS_spread values

# ------------------------------------------------------------  # separator for readability
# 4) Optional: show best/worst months for LS_5050                  # describes optional diagnostic
# ------------------------------------------------------------  # end separator
tmp = LS_5050.dropna().copy()                                   # drop missing months from LS_5050 and copy for safe sorting/min/max
print("\nLS_5050 min/max:", float(tmp.min()), float(tmp.max()))  # print min and max monthly LS_5050 values
print("\nTop 10 LS_5050 months:")                                # print header for top 10 months
display(tmp.sort_values(ascending=False).head(10))               # display 10 best months for LS_5050 (descending)
print("\nBottom 10 LS_5050 months:")                             # print header for bottom 10 months
display(tmp.sort_values(ascending=True).head(10))                # display 10 worst months for LS_5050 (ascending)

# ------------------------------------------------------------  # separator for readability
# (ADDED) Step 7 diagnostics table(s), like Step 9 "gross vs net" # describes added diagnostic describe() tables
# ------------------------------------------------------------  # end separator

# A) Distribution of the strategy return series themselves        # describes diagnostic table A
step7_check = pd.DataFrame({                                     # build a DataFrame holding both LS series (aligned by index)
    "LS_spread": pd.to_numeric(LS_spread, errors="coerce"),       # coerce LS_spread to numeric for clean describe()
    "LS_5050":   pd.to_numeric(LS_5050, errors="coerce"),         # coerce LS_5050 to numeric for clean describe()
})                                                               # end DataFrame constructor
print("\nStep 7 check (LS series distribution):")                # print a label for this diagnostic output
display(step7_check.describe())                                   # display descriptive stats (count/mean/std/min/quantiles/max)

# B) If you still have step6_summary, show month-level long/short + LS  # describes diagnostic table B
if "step6_summary" in globals():                                 # check whether step6_summary exists in the current namespace
    cols = [c for c in ["long_ret_next", "short_ret_next", "LS_spread", "LS_5050"] if c in step6_summary.columns]  # keep only columns that exist
    if cols:                                                     # only proceed if at least one of those columns exists
        print("\nStep 7 check (Top/Bottom month-level inputs):")  # print a label for this diagnostic output
        display(step6_summary[cols].apply(pd.to_numeric, errors="coerce").describe())  # coerce selected cols to numeric then show describe()

# C) If you still have membership, show realized returns BEFORE vs AFTER capping  # describes diagnostic table C
#    (this is the closest analog to "gross vs net" for Strategy 2 Step 6/7)       # explains why this diagnostic is meaningful
if "membership" in globals():                                    # check whether membership exists in the current namespace
    cols = [c for c in ["R_next", "R_next_cap"] if c in membership.columns]  # select which columns exist among R_next and R_next_cap
    if cols:                                                     # only proceed if at least one of those columns exists
        print("\nStep 7 check (selected names: R_next vs R_next_cap):")  # print a label for this diagnostic output
        display(membership[cols].apply(pd.to_numeric, errors="coerce").describe())  # coerce selected cols to numeric then show describe()

# =========================  # section header
# Step 8: Inverse-volatility weighting (Strategy 2: Top-3 / Bottom-3)  # what this step is
# Uses:  # inputs used
#   - step5 (gross position returns) to compute sigma_rt (past vol, no look-ahead)  # vol source
#   - membership (from Step 6) which already has R_next_cap and side  # membership source
# Outputs:  # outputs produced
#   - LS_5050_iv, LS_spread_iv  # IV long-short series
#   - summary8 + plots + histograms (NOW includes LS_spread_iv plots/hists too)  # reporting outputs
# Adds (DIAGNOSTICS to match Strategy 1 Step 8 screenshots):  # what we added
#   - position-level describe(): sigma_rt, w_raw, w_norm, R_next, R_next_cap  # screenshot 1 columns
#   - month-level describe(): loser_bucket_ret_iv, winner_bucket_ret_iv, LS_spread_iv, LS_5050_iv  # screenshot 2 columns
# =========================  # end header

# Guardrails  # sanity checks before running
if "membership" not in globals() or "step5" not in globals():  # make sure required inputs exist
    raise NameError("Step 8 expects `membership` (from Step 6) and `step5` (from Step 5).")  # stop early with clear message

# --- 1) Compute sigma_rt per (permno, month) using ONLY info through t-2 (no look-ahead)  # step 1 description
vol_base = step5[["permno", "month", "R_t1"]].copy()  # base table for volatility: per-stock gross returns
vol_base["permno"] = pd.to_numeric(vol_base["permno"], errors="coerce").astype("Int64")  # standardize permno dtype
vol_base["R_t1"]   = pd.to_numeric(vol_base["R_t1"], errors="coerce")  # ensure returns are numeric
vol_base["month"]  = pd.PeriodIndex(vol_base["month"].astype(str), freq="M")  # ensure month is monthly PeriodIndex
vol_base = vol_base.sort_values(["permno", "month"]).reset_index(drop=True)  # sort before rolling within permno

# sigma_rt at month t: std of (t-12 ... t-2) using shift(2).rolling(11)  # timing convention explanation
vol_base["sigma_rt"] = (  # compute past volatility with no look-ahead
    vol_base.groupby("permno", group_keys=False)["R_t1"]  # within each permno, take its return series
            .apply(lambda s: s.shift(2).rolling(window=11, min_periods=8).std(ddof=1))  # std of R_{t-12}..R_{t-2}
)  # end sigma_rt computation

sigma_table = vol_base[["permno", "month", "sigma_rt"]].copy()  # table to merge onto membership

# --- 2) Merge sigma onto membership  # step 2 description
mem8 = membership.copy()  # working copy so we don't mutate original membership

# Ensure consistent dtypes  # dtype harmonization for merge/groupby
mem8["permno"] = pd.to_numeric(mem8["permno"], errors="coerce").astype("Int64")  # standardize permno dtype
if not isinstance(mem8["month"].dtype, pd.PeriodDtype):  # if month is not PeriodIndex dtype yet
    mem8["month"] = pd.PeriodIndex(mem8["month"].astype(str), freq="M")  # convert month to monthly PeriodIndex

mem8 = mem8.merge(sigma_table, on=["permno", "month"], how="left")  # attach sigma_rt at signal month t

# Inverse-vol weights (drop sigma<=0)  # compute inv-vol and ignore invalid sigmas
mem8["inv_vol"] = np.where(  # inv-vol = 1/sigma where sigma valid
    mem8["sigma_rt"].notna() & (mem8["sigma_rt"] > 0),  # condition: sigma exists and positive
    1.0 / mem8["sigma_rt"],  # inv-vol value
    np.nan  # else missing
)

# Require valid realized next-month return and valid inv_vol  # drop rows that cannot be used in weighting
mem8 = mem8[mem8["R_next_cap"].notna() & mem8["inv_vol"].notna()].copy()  # keep only rows with return + inv-vol

# --- 3) Compute within-month, within-side normalized weights  # normalize weights within (month, side)
mem8["w_raw"] = mem8["inv_vol"]  # (MATCH STRAT 1 naming) raw weight before normalization
w_sum = mem8.groupby(["month", "side"])["w_raw"].transform("sum")  # sum of raw weights per month and side
mem8["w_norm"] = np.where(w_sum > 0, mem8["w_raw"] / w_sum, np.nan)  # (MATCH STRAT 1 naming) normalized weights
mem8["w"] = mem8["w_norm"]  # keep your original downstream column name (so later code stays consistent)

# --- 4) Compute weighted long/short returns per month, then LS  # step 4 description
by_ms = (  # compute side-level IV returns each month
    mem8.groupby(["month", "side"])  # group by month and side (LONG/SHORT)
        .apply(lambda g: float((g["w"] * g["R_next_cap"]).sum()) if g["w"].notna().any() else np.nan)  # weighted sum
        .rename("ret_next_iv")  # name the resulting series
        .reset_index()  # convert to DataFrame
)

wide8 = by_ms.pivot(index="month", columns="side", values="ret_next_iv").sort_index()  # wide table: LONG and SHORT columns

long_iv  = wide8.get("LONG",  pd.Series(index=wide8.index, dtype="float64"))  # long leg IV return series
short_iv = wide8.get("SHORT", pd.Series(index=wide8.index, dtype="float64"))  # short leg IV return series

LS_spread_iv = (long_iv - short_iv).rename("LS_spread_iv")  # long-short spread series (not normalized)
LS_5050_iv   = (0.5 * (long_iv - short_iv)).rename("LS_5050_iv")  # 50/50 self-financing version

# --- 5) Performance summary (Step-7 style)  # compute perf stats using your Step-7 eval_series()
summary8 = pd.DataFrame([  # build summary DataFrame with one row per LS series
    eval_series(LS_5050_iv,   "LS_5050_iv (inv-vol, 50/50)"),  # stats for 50/50 IV series
    eval_series(LS_spread_iv, "LS_spread_iv (inv-vol, spread)")  # stats for spread IV series
])  # end summary DataFrame build

print("\n=== Step 8 Performance Summary (Inverse-Vol) ===")  # header
display(summary8)  # show summary table

# --- 6) Plots + histograms (NOW includes LS_spread_iv too)  # visualization section
plot_cumulative(LS_5050_iv, "LS_5050_iv (inv-vol, 50/50)")  # cumulative plot for LS_5050_iv
plot_hist(LS_5050_iv, "LS_5050_iv (inv-vol, 50/50)", bins=30)  # histogram for LS_5050_iv
plot_cumulative(LS_spread_iv, "LS_spread_iv (inv-vol, spread)")  # cumulative plot for LS_spread_iv
plot_hist(LS_spread_iv, "LS_spread_iv (inv-vol, spread)", bins=30)  # histogram for LS_spread_iv

# ------------------------------------------------------------  # separator
# (ADDED) Step 8 diagnostics table(s), MATCH Strategy 1 Step 8 screenshots  # diagnostics header
# ------------------------------------------------------------  # separator

# 1) Position-level describe(): sigma_rt, w_raw, w_norm, R_next, R_next_cap  # screenshot 1 diagnostics
print("\nStep 8 check (position-level inputs used):")  # print header to match screenshot wording
cols_pos = ["sigma_rt", "w_raw", "w_norm", "R_next", "R_next_cap"]  # desired columns (exactly like Strategy 1 screenshot)
for c in cols_pos:  # ensure all requested columns exist (avoid KeyError if one is missing)
    if c not in mem8.columns:  # if missing
        mem8[c] = np.nan  # create as NaN so describe() still prints with the right columns
display(mem8[cols_pos].apply(pd.to_numeric, errors="coerce").describe())  # show describe table

# 2) Month-level describe(): loser_bucket_ret_iv, winner_bucket_ret_iv, LS_spread_iv, LS_5050_iv  # screenshot 2 diagnostics
step8_check_month = pd.DataFrame({  # build month-level diagnostic DataFrame
    "loser_bucket_ret_iv":  pd.to_numeric(short_iv, errors="coerce"),  # loser = SHORT leg return (top/bottom framework)
    "winner_bucket_ret_iv": pd.to_numeric(long_iv, errors="coerce"),  # winner = LONG leg return
    "LS_spread_iv":         pd.to_numeric(LS_spread_iv, errors="coerce"),  # long - short
    "LS_5050_iv":           pd.to_numeric(LS_5050_iv, errors="coerce"),  # 0.5*(long - short)
})  # end DataFrame construction
print("\nStep 8 check (month-level IV winner/loser + LS):")  # print header to match Strategy 1 style
display(step8_check_month.describe())  # show describe table

# =========================  # section header: describes what this cell is
# Step 9: Net-of-costs returns (Strategy 2) — cost logic identical to Strategy 1 Step 9  # what Step 9 does
# Then re-run Strategy 2 Top-3/Bottom-3 selection on NET returns.  # workflow summary
#  # blank-line separator (comment-only)
# Adds Step-9-style describe() tables to match Strategy 1 Step 9 screenshots:  # what diagnostics we add
#   (A) Position-level economics (expanded): entry_mid, exit_intrinsic, R_t1, entry_eff, exit_eff, roundtrip_cost, R_t1_net  # position-level describe() columns
#   (B) Month-level inputs (net): long_ret_next_net, short_ret_next_net, LS_spread_net, LS_5050_net  # month-level describe() columns
#   (C) Membership-level (net): R_next_net, R_next_net_cap  # membership-level describe() columns
# =========================  # end header block
# -----------------------------  # section divider
# Guardrails  # sanity checks before running
# -----------------------------  # section divider
if "all_monthly" not in globals():  # ensure upstream Step 4/positions table exists
    raise NameError("Step 9 expects `all_monthly` from your earlier steps.")  # stop early with a clear error

# NOTE: This Step 9 also assumes these exist from earlier Strategy 2 cells:  # dependencies needed later
#   - pick_top_bottom_with_fallback(df, N_LONG, N_SHORT)  # function that selects Top/Bottom names robustly
#   - N_LONG, N_SHORT (typically 3 and 3)  # how many longs/shorts per month
#   - eval_series, plot_cumulative, plot_hist (if you keep the evaluation/plots section)  # optional evaluation tools

# -----------------------------  # section divider
# 0) Parameters (same structure as Strategy 1 Step 9)  # cost model parameters
# -----------------------------  # section divider
slippage_rate_entry = 0.001   # 0.10% entry slippage  # you pay above mid when entering
exit_cost_rate      = 0.001   # 0.10% exit haircut  # you receive below intrinsic/exit value
cost_bps_per_leg    = 5       # 5 bps per leg  # per-leg transaction cost in basis points
n_legs              = 2       # call + put  # straddle has two option legs
cost_rate_total     = (cost_bps_per_leg / 10000.0) * n_legs  # convert bps to decimal and multiply by legs

# Keep same caps as Step 6 for apples-to-apples comparison  # ensures comparability across steps
CAP_LO, CAP_HI = -0.95, 5.0  # lower/upper cap for winsorizing option returns

# -----------------------------  # section divider
# 1) Build net-of-costs returns at POSITION level (IDENTICAL to Strategy 1 logic)  # compute R_t1_net
# -----------------------------  # section divider
all_monthly_9 = all_monthly.copy()  # work on a copy to keep original gross table intact

# force numeric on key columns if present  # avoids object/string dtype issues
for c in ["entry_mid", "exit_intrinsic", "R_t1"]:  # iterate through required position-level columns
    if c in all_monthly_9.columns:  # guard in case a column is missing
        all_monthly_9[c] = pd.to_numeric(all_monthly_9[c], errors="coerce")  # invalid parses -> NaN

# valid rows: entry_mid > 0 and exit_intrinsic >= 0  # define where net return is computable
valid = (  # start boolean mask
    all_monthly_9["entry_mid"].notna() & (all_monthly_9["entry_mid"] > 0) &  # entry premium must be positive
    all_monthly_9["exit_intrinsic"].notna() & (all_monthly_9["exit_intrinsic"] >= 0)  # intrinsic payoff is nonnegative
)  # end boolean mask

# execution-adjusted entry (pay worse than mid)  # incorporate entry slippage
all_monthly_9["entry_eff"] = np.where(  # vectorized conditional assignment
    valid,  # only compute on valid positions
    all_monthly_9["entry_mid"] * (1.0 + slippage_rate_entry),  # effective entry paid
    np.nan  # invalid rows -> NaN
)  # end entry_eff assignment

# execution-adjusted exit (receive a haircut vs intrinsic)  # incorporate exit haircut
all_monthly_9["exit_eff"] = np.where(  # vectorized conditional assignment
    valid,  # only compute on valid positions
    all_monthly_9["exit_intrinsic"] * (1.0 - exit_cost_rate),  # effective exit received
    np.nan  # invalid rows -> NaN
)  # end exit_eff assignment

# proportional roundtrip cost based on entry_mid notional  # transaction cost modeled as % of entry premium
all_monthly_9["roundtrip_cost"] = np.where(  # vectorized conditional assignment
    valid,  # only compute on valid positions
    all_monthly_9["entry_mid"] * cost_rate_total,  # total cost across both legs
    np.nan  # invalid rows -> NaN
)  # end roundtrip_cost assignment

# net return uses entry_mid in denominator (to match your gross R_t1 definition scale)  # keep denominator consistent
all_monthly_9["R_t1_net"] = np.where(  # vectorized conditional assignment
    valid,  # only compute on valid positions
    (all_monthly_9["exit_eff"] - all_monthly_9["entry_eff"] - all_monthly_9["roundtrip_cost"]) / all_monthly_9["entry_mid"],  # net return formula
    np.nan  # invalid rows -> NaN
)  # end R_t1_net assignment

# ---- (ADDED) Step 9 check (position-level economics, expanded) ----  # diagnostic table header
print("\nStep 9 check (position-level economics, expanded):")  # print label for the describe() table
pos_cols = ["entry_mid", "exit_intrinsic", "R_t1", "entry_eff", "exit_eff", "roundtrip_cost", "R_t1_net"]  # desired columns
pos_cols = [c for c in pos_cols if c in all_monthly_9.columns]  # keep only columns that exist
display(all_monthly_9[pos_cols].describe())  # show distribution stats for position-level economics

# -----------------------------  # section divider
# 2) Recompute Step 5 momentum on NET returns (same timing rules)  # compute MOM_rt_net from R_t1_net
# -----------------------------  # section divider
step5_net = all_monthly_9.copy()  # start from net position-level table (contains R_t1_net)
step5_net["permno"] = pd.to_numeric(step5_net["permno"], errors="coerce").astype("Int64")  # standardize permno
step5_net["R_t1_net"] = pd.to_numeric(step5_net["R_t1_net"], errors="coerce")  # ensure net return is numeric

step5_net["_month_ts"] = pd.PeriodIndex(step5_net["month"].astype(str), freq="M").to_timestamp()  # month timestamp for sorting/rolling
step5_net = step5_net.sort_values(["permno", "_month_ts"]).reset_index(drop=True)  # sort within permno before rolling

step5_net["MOM_rt_net"] = (  # momentum signal at month t
    step5_net.groupby("permno", group_keys=False)["R_t1_net"]  # group by permno
             .apply(lambda s: s.shift(2).rolling(window=11, min_periods=8).mean())  # mean of t-12..t-2 (exclude t-1), min 8 obs
)  # end MOM_rt_net assignment

step5_net = step5_net.drop(columns=["_month_ts"])  # remove helper timestamp column

# -----------------------------  # section divider
# 3) Strategy 2 Step 6 logic on NET: Top-3/Bottom-3 (reuse your existing fallback function)  # pick Top/Bottom on net momentum
# -----------------------------  # section divider
panel6_net = step5_net[["month", "permno", "R_t1_net", "MOM_rt_net"]].copy()  # panel needed for ranking & next-month returns
panel6_net["permno"] = pd.to_numeric(panel6_net["permno"], errors="coerce").astype("Int64")  # standardize permno dtype
panel6_net["R_t1_net"] = pd.to_numeric(panel6_net["R_t1_net"], errors="coerce")  # ensure numeric
panel6_net["MOM_rt_net"] = pd.to_numeric(panel6_net["MOM_rt_net"], errors="coerce")  # ensure numeric
panel6_net["month"] = pd.PeriodIndex(panel6_net["month"].astype(str), freq="M")  # ensure PeriodIndex months

panel6_net = panel6_net.sort_values(["permno", "month"]).reset_index(drop=True)  # sort to compute shift(-1) correctly
panel6_net["R_next_net"] = panel6_net.groupby("permno")["R_t1_net"].shift(-1)  # realized next-month net return (t+1)
panel6_net["R_next_net_cap"] = panel6_net["R_next_net"].clip(lower=CAP_LO, upper=CAP_HI)  # cap extreme net returns

# keep only rows where we can rank on MOM_rt_net (you can still have missing R_next_net later)  # must have signal to sort
panel6_net = panel6_net.dropna(subset=["permno", "month", "MOM_rt_net"]).copy()  # drop rows without momentum signal

rows9 = []  # will collect month-level summary rows
members9 = []  # will collect month-level membership tables

for m, df_m in panel6_net.groupby("month", sort=True):  # iterate month-by-month (sorted) for selection
    # adapt column names to what pick_top_bottom_with_fallback expects  # rename to the function’s expected schema
    df_tmp = df_m.rename(columns={  # rename columns for compatibility
        "MOM_rt_net": "MOM_rt",  # signal column expected by picker
        "R_next_net": "R_next",  # next-month return column expected by picker
        "R_next_net_cap": "R_next_cap"  # capped next-month return column expected by picker
    }).copy()  # work on a copy to avoid mutating df_m

    long_ret, short_ret, mem = pick_top_bottom_with_fallback(df_tmp, N_LONG, N_SHORT)  # select Top/Bottom and get realized returns + membership

    rows9.append({  # append one month summary record
        "month": m,  # month identifier
        "n_long": int(mem[mem["side"] == "LONG"].shape[0]),  # number of long names selected
        "n_short": int(mem[mem["side"] == "SHORT"].shape[0]),  # number of short names selected
        "long_ret_next_net": long_ret,  # average (or fallback) next-month net return for longs
        "short_ret_next_net": short_ret,  # average (or fallback) next-month net return for shorts
        "LS_spread_net": (long_ret - short_ret) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # winner-loser spread (net)
        "LS_5050_net": (0.5 * (long_ret - short_ret)) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # 50/50 self-financing return (net)
    })  # end month summary dict

    # keep membership table (net names)  # preserve selection constituents for diagnostics
    mem_keep = mem[["month", "permno", "MOM_rt", "R_next", "R_next_cap", "side"]].copy()  # keep core columns
    mem_keep = mem_keep.rename(columns={  # rename to net-labeled columns
        "MOM_rt": "MOM_rt_net",  # net momentum label
        "R_next": "R_next_net",  # net realized next-month return label
        "R_next_cap": "R_next_net_cap"  # net capped realized return label
    })  # end rename
    members9.append(mem_keep)  # store month membership block

step9_summary = pd.DataFrame(rows9).sort_values("month").reset_index(drop=True)  # build month-level summary table
membership_net = pd.concat(members9, ignore_index=True) if len(members9) else pd.DataFrame()  # concatenate membership rows (or empty df)

LS_spread_net = step9_summary.set_index("month")["LS_spread_net"]  # extract spread series indexed by month
LS_5050_net   = step9_summary.set_index("month")["LS_5050_net"]  # extract 50/50 series indexed by month

print("\n=== Step 9 (Net) Top/Bottom 3 Summary ===")  # print label
display(step9_summary.head(10))  # show first 10 months of summary
display(membership_net.head(20))  # show first 20 membership rows for inspection

# ---- (ADDED) Step 9 check (month-level inputs, net) ----  # diagnostic header
print("\nStep 9 check (month-level inputs, net):")  # print label for month-level describe
ml_cols = ["long_ret_next_net", "short_ret_next_net", "LS_spread_net", "LS_5050_net"]  # desired month-level columns
ml_cols = [c for c in ml_cols if c in step9_summary.columns]  # keep only existing columns
display(step9_summary[ml_cols].apply(pd.to_numeric, errors="coerce").describe())  # show describe() for month-level inputs

# ---- (ADDED) Step 9 check (membership-level inputs, net) ----  # diagnostic header
if isinstance(membership_net, pd.DataFrame) and (not membership_net.empty):  # only run if membership table exists and is non-empty
    mem_cols = ["R_next_net", "R_next_net_cap"]  # desired membership-level columns
    mem_cols = [c for c in mem_cols if c in membership_net.columns]  # keep only existing columns
    if len(mem_cols) > 0:  # ensure we have something to describe
        print("\nStep 9 check (membership-level inputs, net):")  # print label for membership-level describe
        display(membership_net[mem_cols].apply(pd.to_numeric, errors="coerce").describe())  # show describe() for membership-level inputs

# -----------------------------  # section divider
# 4) Step 7-style evaluation (net) + plots/histograms (unchanged)  # performance summary + plots
# -----------------------------  # section divider
summary9 = pd.DataFrame([  # build performance summary table for LS series
    eval_series(LS_5050_net,   "LS_5050_net (Top3/Bottom3, net)"),  # eval 50/50 net series
    eval_series(LS_spread_net, "LS_spread_net (Top3/Bottom3, spread, net)")  # eval spread net series
])  # end DataFrame construction

print("\n=== Step 9 Net Performance Summary ===")  # print label for summary table
display(summary9)  # show summary metrics table

plot_cumulative(LS_5050_net, "LS_5050_net (Top3/Bottom3, net)")  # plot cumulative curves for 50/50 net series
plot_hist(LS_5050_net, "LS_5050_net (Top3/Bottom3, net)", bins=30)  # histogram for 50/50 net series

plot_cumulative(LS_spread_net, "LS_spread_net (Top3/Bottom3, spread, net)")  # plot cumulative curves for spread net series
plot_hist(LS_spread_net, "LS_spread_net (Top3/Bottom3, spread, net)", bins=30)  # histogram for spread net series
