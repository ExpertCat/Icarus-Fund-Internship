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


from pathlib import Path
import pandas as pd

print("BASE_DIR  =", BASE_DIR,  "exists:", Path(BASE_DIR).exists())
print("MAP_PATH  =", MAP_PATH,  "exists:", Path(MAP_PATH).exists())
print("STOCK_PATH=", STOCK_PATH,"exists:", Path(STOCK_PATH).exists())

# show year folders
if Path(BASE_DIR).exists():
    years = sorted([p.name for p in Path(BASE_DIR).iterdir() if p.is_dir()])[:10]
    print("Year folders under BASE_DIR (first 10):", years)

# test one formation file path
def third_fridays(start="2018-01-01", end="2018-12-31"):
    months = pd.date_range(start=start, end=end, freq="MS")
    out = []
    for m in months:
        fridays = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")
        out.append(fridays[2])
    return pd.DatetimeIndex(out)

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


# =========================
# Step 6: Form Momentum Portfolios (Q buckets) and compute long-short
# Correct timing: sort on MOM_{t}, evaluate R_{t+1}
# Requires: step5 DataFrame with columns at least ["month","permno","R_t1","MOM_rt"]
# =========================
# What this step is doing conceptually:
# - At each month t, you compute MOM_{r,t} for each stock r (already done in Step 5).
# - You rank stocks by MOM_{r,t} and split them into Q groups (quantile buckets).
# - You then measure each group’s performance using NEXT MONTH return R_{r,t+1}.
# - The “momentum” long-short return is: winners bucket (highest MOM) minus losers bucket (lowest MOM).

Q = 5  # Number of quantile buckets: 5=quintiles, 10=deciles, 3=terciles
# Higher Q = more granularity but less names per bucket (more noisy with only ~20 names).

panel6 = step5[["month", "permno", "R_t1", "MOM_rt"]].copy()
# Create a working panel for Step 6 containing only what we need:
# - month: evaluation month t
# - permno: stock identifier
# - R_t1: the “realized return” series used in Step 5 and for shifting to make R_{t+1}
# - MOM_rt: the signal used for sorting at month t
# .copy() avoids unintended changes to step5.

# -------------------------
# Standardize types
# -------------------------
panel6["permno"] = pd.to_numeric(panel6["permno"], errors="coerce").astype("Int64")
# Convert permno to numeric. Bad values become NaN (then drop later).
# Use nullable Int64 because some rows may be missing.

panel6["R_t1"]   = pd.to_numeric(panel6["R_t1"], errors="coerce")
# Ensure R_t1 is float-like; strings/invalid -> NaN.

panel6["MOM_rt"] = pd.to_numeric(panel6["MOM_rt"], errors="coerce")
# Ensure MOM_rt is numeric; invalid -> NaN.

panel6["month"]  = pd.PeriodIndex(panel6["month"].astype(str), freq="M")
# Convert month into pandas PeriodIndex with monthly frequency.
# This gives correct chronological ordering and allows easy month-to-timestamp conversion for plots.

# -------------------------
# 1) Build next-month return R_{t+1} within each permno
# -------------------------
panel6 = panel6.sort_values(["permno", "month"]).reset_index(drop=True)
# IMPORTANT: We must sort by permno and time before shifting, otherwise "next month" is wrong.
# reset_index(drop=True) makes the index clean after sorting.

panel6["R_next"] = panel6.groupby("permno")["R_t1"].shift(-1)
# For each permno (each stock):
# - shift(-1) moves the next row’s R_t1 up.
# - After this, the row at month t contains R_next = R_{t+1}.
#
# Timing alignment:
# - MOM_rt is MOM_{t} (computed from older history in Step 5).
# - We will evaluate that ranking using R_next = R_{t+1} (the next month’s realized return).

# -------------------------
# 2) At month t, we need MOM_t and R_{t+1}
# -------------------------
panel6 = panel6.dropna(subset=["permno", "month", "MOM_rt", "R_next"]).copy()
# Keep only rows where:
# - permno exists
# - month exists
# - MOM_rt exists (signal available)
# - R_next exists (we can evaluate next-month performance)
#
# This also automatically drops:
# - early months with insufficient history for MOM
# - last month for each permno where shift(-1) has no next return

# -------------------------
# 2.5) Stabilize extreme option returns (critical for short leg)
# -------------------------
# Rationale:
# - Option returns can be extremely large positive or negative (especially if you are modeling "short" exposure).
# - Even if your strategy is not literally shorting options, the cross-sectional long-short return
#   can be dominated by a single extreme month/stock.
# - Clipping ("winsorization") is a *portfolio mechanics* stabilization step.
#   You should report it, because it changes the data you use for portfolio returns.
CAP_LO, CAP_HI = -0.95, 5.00  # lower and upper caps; tune; report in writeup
# CAP_LO = -0.95 means the worst allowed return is -95% (i.e., cannot lose more than 95% in this capped series).
# CAP_HI = 5.00 means max allowed return is +500% in a month.
# These are not "economic truths"; they are a modeling guardrail for aggregation.

panel6["R_next_cap"] = panel6["R_next"].clip(lower=CAP_LO, upper=CAP_HI)
# clip replaces:
# - any R_next < CAP_LO with CAP_LO
# - any R_next > CAP_HI with CAP_HI
# - values inside the range are unchanged
#
# IMPORTANT:
# - Your true realized R_next is still stored in panel6["R_next"].
# - R_next_cap is the modified series used to compute group averages and LS returns.

# -------------------------
# 3) Assign buckets within each month based on MOM_t
# -------------------------
def bucketize_mom(s: pd.Series, q: int) -> pd.Series:
    # This function takes a vector of MOM values for ONE month (across stocks)
    # and outputs a bucket label 1..Q per stock.

    # rank(method="first") avoids qcut failure from duplicates
    r = s.rank(method="first", ascending=True)
    # Why rank first?
    # - pd.qcut can fail or create uneven bins if there are many ties (duplicate MOM values).
    # - Ranking forces a strict ordering (ties are broken by appearance order).
    # ascending=True means smaller MOM gets smaller rank (so bucket 1 becomes "losers").

    # If too few names, return NA
    if r.size < q:
        return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")
    # If you have fewer than Q stocks in that month, you cannot form Q buckets.

    # qcut can still drop bins if not enough unique values; handle that robustly
    try:
        b = pd.qcut(r, q=q, labels=False, duplicates="drop")
        # pd.qcut partitions the ranked values into q equal-count bins.
        # labels=False => bins labeled 0..(q-1).
        # duplicates="drop" => if boundaries are not unique, bins may be reduced.

        # if fewer than q bins were created, treat as unusable for that month
        if pd.Series(b).nunique(dropna=True) < q:
            return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")
        # Guardrail:
        # - If qcut had to drop bins, you no longer have all buckets 1..Q.
        # - For strict "top bucket vs bottom bucket" logic, you might want to skip such months.

        return pd.Series((b + 1).astype("int64"), index=s.index, dtype="Int64")  # 1..Q
        # Convert 0..(q-1) to 1..Q, consistent with your wide[Q] and wide[1] access later.

    except Exception:
        return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")
    # If qcut fails for any reason (e.g., all values identical), return NA buckets for this month.

panel6["bucket"] = panel6.groupby("month")["MOM_rt"].transform(lambda s: bucketize_mom(s, Q))
# For each month:
# - take MOM_rt across all permnos
# - run bucketize_mom
# transform returns a Series aligned to the original rows (same length as panel6).

panel6 = panel6.dropna(subset=["bucket"]).copy()
# Remove months where bucketization failed (bucket is NA for those rows).

panel6["bucket"] = panel6["bucket"].astype("int64")
# Convert bucket to a plain integer (1..Q). After dropping NAs this is safe.

# -------------------------
# 4) Group returns in month t+1 (i.e. average of R_next)
# -------------------------
group_returns = (
    panel6.groupby(["month", "bucket"])                  # group by evaluation month t and bucket assignment at t
          .agg(
               n=("permno", "size"),                     # how many stocks fell into that bucket
               group_ret_next=("R_next_cap", "mean")     # average next-month return for that bucket
          )
          .reset_index()                                 # flatten groupby result back into a DataFrame
          .sort_values(["month", "bucket"])              # sort for readability
)
# Interpretation:
# - For each month t and bucket j:
#   group_ret_next(t, j) = average_{r in bucket j at month t} [ R_{r,t+1} (capped) ]
#
# This is exactly what the document step describes: group-wise returns at t+1.

# -------------------------
# 5) Long-short spread and 50/50 self-financing version
# -------------------------
wide = group_returns.pivot(index="month", columns="bucket", values="group_ret_next").sort_index()
# Convert long format (month,bucket,value) into wide format:
# - rows: months t
# - columns: bucket 1..Q
# - cell: group_ret_next(t, bucket)
#
# Then sort_index() so months are in chronological order.

LS_spread = (wide[Q] - wide[1]).rename("LS_spread")
# Long-short "spread" at each month t:
# - winners bucket (highest MOM: bucket Q)
# - minus losers bucket (lowest MOM: bucket 1)
#
# NOTE: This is NOT a capital-normalized portfolio return unless you explicitly define capital.
# It is a difference in average returns.

LS_5050   = (0.5 * (wide[Q] - wide[1])).rename("LS_5050")
# 50/50 self-financing return:
# - Long $0.5 notional in winners bucket
# - Short $0.5 notional in losers bucket
# Return on $1 collateral = 0.5*R_winners - 0.5*R_losers = 0.5*(R_winners - R_losers)
#
# This IS a more interpretable “portfolio return” series than LS_spread
# (still simplified because we’re averaging within buckets).

# -------------------------
# 6) Cumulative plots
# -------------------------
cum_spread = LS_spread.cumsum()     # additive cumulative sum of monthly spreads
# Additive sum: cum(t) = Σ_{u<=t} LS_spread(u)
# This is a "PnL-like" display (not compounding).

cum_5050_add = LS_5050.cumsum()     # additive cumulative sum of monthly 50/50 returns
# This is safe even if LS_5050 <= -1 in some month, because we are not multiplying wealth.
# It's “running sum of monthly returns.”

print("=== Step 6 Summary ===")
print("Months in LS_spread:", int(LS_spread.notna().sum()))
# Count how many months have a valid LS_spread.

print("LS_spread mean/std:", float(LS_spread.mean()), float(LS_spread.std()))
# Mean and standard deviation of LS_spread (includes NaNs automatically ignored by pandas).

print("LS_spread Sharpe (WARNING: spread, not a portfolio return):",
      float(LS_spread.mean()/LS_spread.std()) if LS_spread.std() != 0 else np.nan)
# Computes "Sharpe-like" ratio for LS_spread.
# Warning is correct:
# - Sharpe is typically defined on a portfolio return series.
# - LS_spread is a difference of average returns; scaling/capital interpretation is ambiguous.

print("LS_5050 mean/std:", float(LS_5050.mean()), float(LS_5050.std()))
# Mean and standard deviation of the 50/50 portfolio return series.

print("LS_5050 Sharpe (monthly):",
      float(LS_5050.mean()/LS_5050.std()) if LS_5050.std() != 0 else np.nan)
# Standard monthly Sharpe estimate (risk-free ignored).

# Plot 1: cumulative spread PnL
plt.figure()                                                 # start a new figure
plt.plot(cum_spread.index.to_timestamp(), cum_spread.values)  # x-axis: month timestamps, y-axis: cumulative sum
plt.title(f"Cumulative Long-Short Spread PnL (Q={Q})")        # plot title with Q displayed
plt.xlabel("Month")                                          # x-axis label
plt.ylabel("Cumulative PnL (sum of monthly spreads)")         # y-axis label describing additive nature
plt.show()                                                   # render the plot

# Plot 2: cumulative 50/50 additive PnL
plt.figure()                                                     # start a new figure
plt.plot(cum_5050_add.index.to_timestamp(), cum_5050_add.values)  # plot cumulative additive 50/50 return
plt.title(f"Cumulative 50/50 Long-Short (additive) (Q={Q})")      # title
plt.xlabel("Month")                                              # x-axis label
plt.ylabel("Cumulative sum")                                      # y-axis label (additive)
plt.show()                                                       # render the plot

# Deliverable table: membership at month t with next-month return used for evaluation
membership = panel6[["month","permno","MOM_rt","bucket","R_next"]].copy()
# membership records, for each month t:
# - which permnos were eligible
# - their MOM signal
# - which bucket they were placed into
# - their realized next-month return R_next (UNcapped; you kept the raw here)
#
# This matches typical deliverables: you can list QH/Q L constituents each month.

# Convenience outputs
QH = membership[membership["bucket"] == Q].copy()
# QH = winners bucket constituents (top momentum group)

QL = membership[membership["bucket"] == 1].copy()
# QL = losers bucket constituents (bottom momentum group)

# =========================  # section header (purely informational)
# Step 7: Evaluate Strategy Performance (Strategy 1)  # describes what this cell is for (Strategy 1 evaluation)
# Uses:  # lists required inputs
#   - LS_spread  # long-short spread series from Step 6 (winner bucket - loser bucket)
#   - LS_5050  # 50/50 self-financing series derived from LS_spread (0.5*(winner-loser))
# Adds:  # lists additions this step makes beyond standard Step 7
#   - Step-9-style describe() diagnostic tables  # extra describe() tables for debugging distributions
#   - A Strategy-1 equivalent of "long_ret_next / short_ret_next" month-level table  # reconstruct winner/loser bucket inputs
#     (winner bucket vs loser bucket)  # clarifies what long/short mean for Strategy 1 buckets
# =========================  # end header block

def eval_series(R: pd.Series, name: str) -> dict:  # define function that computes summary performance stats for a return series
    R = R.dropna().copy()  # drop NaNs and copy to avoid modifying the original series
    R = pd.to_numeric(R, errors="coerce").dropna()  # force numeric (bad parses -> NaN) and drop NaNs again

    T = int(R.shape[0])  # number of valid observations (months)
    mean_m = float(R.mean())  # average monthly return
    vol_m  = float(R.std(ddof=1))  # monthly volatility: sample standard deviation (ddof=1)
    sharpe_m = mean_m / vol_m if vol_m > 0 else np.nan  # monthly Sharpe ratio (rf assumed 0); NaN if vol=0
    tstat = mean_m / (vol_m / np.sqrt(T)) if (vol_m > 0 and T > 1) else np.nan  # t-stat of mean return vs 0

    ann_ret = 12.0 * mean_m  # annualized return via linear scaling (12 * monthly mean)
    ann_vol = np.sqrt(12.0) * vol_m  # annualized volatility via sqrt-time scaling (sqrt(12) * monthly std)
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan  # annualized Sharpe ratio

    can_compound = bool(((1.0 + R) > 0).all())  # True if all months satisfy (1+R)>0 so cumprod is valid

    return {  # return a dict of computed metrics (to later convert into a DataFrame row)
        "name": name,  # label for this series in the output table
        "T": T,  # number of months used
        "mean_monthly": mean_m,  # monthly mean
        "vol_monthly": vol_m,  # monthly volatility
        "sharpe_monthly": sharpe_m,  # monthly Sharpe
        "tstat": tstat,  # t-stat of mean
        "ann_ret": ann_ret,  # annualized mean (linear)
        "ann_vol": ann_vol,  # annualized volatility
        "ann_sharpe": ann_sharpe,  # annualized Sharpe
        "min": float(R.min()) if T > 0 else np.nan,  # worst month value (if any observations exist)
        "max": float(R.max()) if T > 0 else np.nan,  # best month value (if any observations exist)
        "can_compound": can_compound,  # flag indicating whether geometric compounding is defined
    }  # end dict

def plot_cumulative(R: pd.Series, title: str):  # define function to plot cumulative performance for a series
    R = R.dropna().copy()  # drop NaNs and copy (avoid modifying original series)

    if isinstance(R.index, pd.PeriodIndex):  # if index is a PeriodIndex (monthly periods)
        x = R.index.to_timestamp()  # convert PeriodIndex -> TimestampIndex for matplotlib x-axis
    else:  # otherwise treat the index as datetime-like
        x = pd.to_datetime(R.index)  # convert index to datetime for plotting

    can_compound = bool(((1.0 + R) > 0).all())  # determine if compounded curve is mathematically valid

    plt.figure()  # create a new figure for additive cumulative sum plot
    plt.plot(x, R.cumsum().values)  # plot additive cumulative sum (PnL-like, not wealth)
    plt.title(title + " — Cumulative (Additive Sum)")  # set plot title for additive cumulative sum
    plt.xlabel("Month")  # label x-axis
    plt.ylabel("Cumulative sum")  # label y-axis
    plt.show()  # render the figure

    if can_compound:  # if compounding is valid (no month with R <= -1)
        cumprod = (1.0 + R).cumprod() - 1.0  # compute compounded cumulative return = prod(1+R) - 1
        plt.figure()  # create a new figure for compounded cumulative return plot
        plt.plot(x, cumprod.values)  # plot compounded cumulative return
        plt.title(title + " — Cumulative (Compounded)")  # set plot title for compounded curve
        plt.xlabel("Month")  # label x-axis
        plt.ylabel("Cumulative return")  # label y-axis
        plt.show()  # render the compounded plot
    else:  # if compounding is invalid
        print(f"WARNING: Cannot compound {title} because some months have (1+R) <= 0.")  # warn user why no compounding plot

def plot_hist(R: pd.Series, title: str, bins: int = 30):  # define function to plot histogram of series values
    R = pd.to_numeric(R, errors="coerce").dropna()  # coerce to numeric, drop NaNs
    plt.figure()  # create a new histogram figure
    plt.hist(R.values, bins=bins)  # plot histogram of monthly values
    plt.title("Histogram — " + title)  # set histogram title
    plt.xlabel("Monthly return")  # label x-axis
    plt.ylabel("Count")  # label y-axis
    plt.show()  # render the histogram

# ------------------------------------------------------------  # visual separator for section
# Guardrails  # section title: basic sanity checks before using variables
# ------------------------------------------------------------  # visual separator for section
if "LS_5050" not in globals() or "LS_spread" not in globals():  # ensure Step 6 created LS_5050 and LS_spread
    raise NameError("Step 7 expects LS_5050 and LS_spread from your Step 6 cell.")  # stop early if missing

# Infer Q for Strategy 1 (bucket count) if possible  # attempt to infer number of buckets Q from membership data
Q = None  # initialize Q unknown (used to define winner bucket = Q and loser bucket = 1)
if "membership" in globals() and "bucket" in membership.columns:  # check membership exists and has bucket column
    q_val = pd.to_numeric(membership["bucket"], errors="coerce").max()  # infer Q as max bucket label
    if pd.notna(q_val):  # if Q was successfully inferred (not NaN)
        Q = int(q_val)  # convert Q to integer

# ------------------------------------------------------------  # section separator
# (1) Step-9-style diagnostic: LS series distribution  # describe() check for LS series themselves
# ------------------------------------------------------------  # section separator
step7_check = pd.DataFrame({  # create a small DataFrame for describe() on LS series
    "LS_spread": pd.to_numeric(LS_spread, errors="coerce"),  # coerce LS_spread to numeric
    "LS_5050":   pd.to_numeric(LS_5050, errors="coerce"),  # coerce LS_5050 to numeric
})  # end DataFrame constructor
print("\nStep 7 check (LS series distribution):")  # print header for the diagnostic table
display(step7_check.describe())  # show describe() stats for LS series (count/mean/std/min/quantiles/max)

# ------------------------------------------------------------  # section separator
# (2) Strategy-1 equivalent of "Top/Bottom inputs" table  # build winner-bucket vs loser-bucket monthly inputs like Strategy 2
#     long_ret_next  := winner bucket (Q) return  # define long_ret_next as winner bucket return
#     short_ret_next := loser bucket (1) return  # define short_ret_next as loser bucket return
#  # blank comment line to keep “comment on every line” while preserving your structure
# We try to reconstruct this from whatever you have available:  # explains fallback logic for constructing the table
#   Priority A: wide (your Step 6 wide bucket return table)  # if you have wide (month x bucket returns), use it
#   Priority B: group_returns (long form month/bucket returns)  # else try long-form month/bucket return table
#   Priority C: membership (compute mean R_next_cap per month/bucket)  # else compute from membership holdings/returns
# ------------------------------------------------------------  # end section header

step7_inputs = None  # initialize; will become a DataFrame with long_ret_next/short_ret_next/LS columns if we can build it

# ---- A) If wide exists and looks like a bucket-return panel  # first attempt: use wide table if available
if "wide" in globals() and isinstance(wide, pd.DataFrame) and (Q is not None):  # require wide exists, is DataFrame, and Q known
    if (1 in wide.columns) and (Q in wide.columns):  # ensure winner bucket (Q) and loser bucket (1) exist as columns
        long_ret_next  = pd.to_numeric(wide[Q], errors="coerce")  # extract winner bucket monthly returns and coerce numeric
        short_ret_next = pd.to_numeric(wide[1], errors="coerce")  # extract loser bucket monthly returns and coerce numeric
        step7_inputs = pd.DataFrame({  # create month-level input table (like your Strategy 2 screenshot)
            "long_ret_next": long_ret_next,  # store winner bucket returns
            "short_ret_next": short_ret_next,  # store loser bucket returns
        })  # end DataFrame constructor
        step7_inputs["LS_spread"] = step7_inputs["long_ret_next"] - step7_inputs["short_ret_next"]  # recompute LS spread
        step7_inputs["LS_5050"]  = 0.5 * step7_inputs["LS_spread"]  # compute 50/50 series from spread

# ---- B) If group_returns exists (month,bucket -> group_ret_next)  # second attempt: use group_returns if wide not available
if step7_inputs is None and "group_returns" in globals() and (Q is not None):  # only run if A failed and Q known and group_returns exists
    gr = group_returns.copy()  # copy to avoid mutating original group_returns object
    if {"month","bucket"}.issubset(gr.columns):  # ensure it has month and bucket columns
        # try common return column names  # we don't know what you named the return column, so search common names
        ret_col = None  # initialize return column name to None
        for c in ["group_ret_next", "ret_next", "group_ret", "ret"]:  # candidate column names for returns
            if c in gr.columns:  # if candidate exists in the DataFrame
                ret_col = c  # set the return column name
                break  # stop searching once found
        if ret_col is not None:  # proceed only if we found a valid return column
            gr["month"]  = pd.PeriodIndex(gr["month"].astype(str), freq="M")  # normalize month to PeriodIndex for stable pivot/sort
            gr["bucket"] = pd.to_numeric(gr["bucket"], errors="coerce").astype("Int64")  # normalize bucket labels to Int64
            gr[ret_col]  = pd.to_numeric(gr[ret_col], errors="coerce")  # ensure returns are numeric

            wide_tmp = gr.pivot(index="month", columns="bucket", values=ret_col).sort_index()  # pivot into wide month x bucket table
            if (1 in wide_tmp.columns) and (Q in wide_tmp.columns):  # ensure required buckets exist
                long_ret_next  = pd.to_numeric(wide_tmp[Q], errors="coerce")  # winner bucket return series
                short_ret_next = pd.to_numeric(wide_tmp[1], errors="coerce")  # loser bucket return series
                step7_inputs = pd.DataFrame({  # build the input table
                    "long_ret_next": long_ret_next,  # store winner bucket
                    "short_ret_next": short_ret_next,  # store loser bucket
                })  # end DataFrame constructor
                step7_inputs["LS_spread"] = step7_inputs["long_ret_next"] - step7_inputs["short_ret_next"]  # compute LS spread
                step7_inputs["LS_5050"]  = 0.5 * step7_inputs["LS_spread"]  # compute 50/50 LS

# ---- C) Fall back: compute from membership (mean within month/bucket of R_next_cap)  # third attempt: build from membership table
if step7_inputs is None and "membership" in globals() and (Q is not None):  # only run if previous attempts failed and membership exists
    mem = membership.copy()  # copy membership so we can add/calc columns without side effects
    if "R_next_cap" not in mem.columns:  # if capped returns not already present
        if "R_next" in mem.columns:  # if raw next-month returns exist
            CAP_LO, CAP_HI = -0.95, 5.0  # set capping bounds consistent with your pipeline
            mem["R_next"] = pd.to_numeric(mem["R_next"], errors="coerce")  # coerce R_next to numeric
            mem["R_next_cap"] = mem["R_next"].clip(lower=CAP_LO, upper=CAP_HI)  # create capped version
        else:  # if raw returns are missing, cannot compute capped returns
            mem["R_next_cap"] = np.nan  # create NaN column to avoid crashing later

    mem["month"]  = pd.PeriodIndex(mem["month"].astype(str), freq="M")  # normalize month to PeriodIndex
    mem["bucket"] = pd.to_numeric(mem["bucket"], errors="coerce").astype("Int64")  # normalize bucket to Int64
    mem["R_next_cap"] = pd.to_numeric(mem["R_next_cap"], errors="coerce")  # ensure capped returns are numeric

    by_mb = (  # compute month-by-bucket mean returns (equal-weight) from membership
        mem.dropna(subset=["month","bucket","R_next_cap"])  # keep only valid month/bucket/return rows
           .groupby(["month","bucket"])["R_next_cap"]  # group by month and bucket and select return
           .mean()  # average within each month-bucket
           .unstack("bucket")  # convert bucket level into columns (wide format)
           .sort_index()  # sort by month
    )  # end construction

    if (1 in by_mb.columns) and (Q in by_mb.columns):  # if both loser and winner buckets exist
        long_ret_next  = pd.to_numeric(by_mb[Q], errors="coerce")  # winner bucket mean return per month
        short_ret_next = pd.to_numeric(by_mb[1], errors="coerce")  # loser bucket mean return per month
        step7_inputs = pd.DataFrame({  # build the month-level inputs table
            "long_ret_next": long_ret_next,  # store winner bucket series
            "short_ret_next": short_ret_next,  # store loser bucket series
        })  # end DataFrame constructor
        step7_inputs["LS_spread"] = step7_inputs["long_ret_next"] - step7_inputs["short_ret_next"]  # compute LS spread
        step7_inputs["LS_5050"]  = 0.5 * step7_inputs["LS_spread"]  # compute 50/50 LS

# Print the screenshot-style table if we managed to build it  # now show month-level winner/loser table if constructed
if step7_inputs is not None:  # if reconstruction succeeded
    print("\nStep 7 check (Strategy 1 month-level inputs: winner bucket vs loser bucket):")  # header print
    display(step7_inputs.describe())  # show describe() stats for month-level inputs and LS calculations
else:  # if reconstruction failed
    print("\nNOTE: Could not build long_ret_next/short_ret_next table (missing wide/group_returns/membership with buckets).")  # explain why

# ------------------------------------------------------------  # section separator
# (3) Membership diagnostic: R_next vs R_next_cap distribution  # show raw vs capped next-month return distributions (position/member level)
# ------------------------------------------------------------  # section separator
if "membership" in globals() and ("R_next" in membership.columns):  # only run if membership exists and has R_next column
    CAP_LO, CAP_HI = -0.95, 5.0  # define capping bounds for diagnostic consistency
    mem_chk = membership.copy()  # copy membership so we don’t mutate it
    mem_chk["R_next"] = pd.to_numeric(mem_chk["R_next"], errors="coerce")  # coerce R_next to numeric
    mem_chk["R_next_cap"] = mem_chk["R_next"].clip(lower=CAP_LO, upper=CAP_HI)  # compute capped return column

    print("\nStep 7 check (membership: R_next vs R_next_cap):")  # header print
    display(mem_chk[["R_next", "R_next_cap"]].describe())  # show describe() for raw and capped next-month returns

# ------------------------------------------------------------  # section separator
# Evaluate both series  # compute and display performance summary metrics for LS series
# ------------------------------------------------------------  # section separator
stats_5050   = eval_series(LS_5050,   "LS_5050 (50/50 self-financing)")  # compute stats for 50/50 long-short return series
stats_spread = eval_series(LS_spread, "LS_spread (winners - losers spread)")  # compute stats for raw spread series
summary = pd.DataFrame([stats_5050, stats_spread])  # assemble both stats dicts into a DataFrame (two rows)

print("\n=== Step 7 Summary Table ===")  # print header for performance summary
display(summary)  # display performance summary DataFrame

# ------------------------------------------------------------  # section separator
# Plots + histograms  # visualize time-series cumulative behavior and distribution of returns
# ------------------------------------------------------------  # section separator
plot_cumulative(LS_5050, "LS_5050 (50/50 self-financing)")  # plot additive and, if valid, compounded cumulative for LS_5050
plot_hist(LS_5050, "LS_5050 (50/50 self-financing)", bins=30)  # histogram for LS_5050

plot_cumulative(LS_spread, "LS_spread (spread, not normalized)")  # plot additive and, if valid, compounded cumulative for LS_spread
plot_hist(LS_spread, "LS_spread (spread, not normalized)", bins=30)  # histogram for LS_spread

# ------------------------------------------------------------  # section separator
# Optional: best/worst months for LS_5050  # show best and worst months in LS_5050 for quick sanity checking
# ------------------------------------------------------------  # section separator
tmp = pd.to_numeric(LS_5050, errors="coerce").dropna().copy()  # create clean numeric series for sorting (drop NaNs)
print("\nLS_5050 min/max:", float(tmp.min()), float(tmp.max()))  # print min and max monthly returns for LS_5050

print("\nTop 10 LS_5050 months:")  # header for top months table
display(tmp.sort_values(ascending=False).head(10))  # display top 10 months by LS_5050 return

print("\nBottom 10 LS_5050 months:")  # header for bottom months table
display(tmp.sort_values(ascending=True).head(10))  # display bottom 10 months by LS_5050 return

# =========================                                 # section header (comment only)
# Step 8: Inverse-vol weighting (Strategy 1, SAME membership as Step 6)  # describe what this step does
#                                                      # spacer comment
# Requires:                                             # list required inputs
#   - step5 with ["month","permno","R_t1"]              # Step 5 output used to compute sigma_rt (past vol)
#   - membership with ["month","permno","bucket","R_next"]  # Step 6 membership used to weight and evaluate
#                                                      # spacer comment
# Adds:                                                 # list additional outputs/checks
#   - Step-9-style describe() diagnostic tables         # extra sanity-check tables
# =========================                                 # section footer (comment only)

# ------------------------------------------------------------  # visual separator for readability
# Guardrails                                                     # validate prerequisites before running
# ------------------------------------------------------------  # visual separator for readability
if "step5" not in globals() or "membership" not in globals():  # verify required inputs exist in the notebook namespace
    raise NameError("Step 8 expects `step5` (from Step 5) and `membership` (from Step 6).")  # stop early with a clear error

if "bucket" not in membership.columns or "R_next" not in membership.columns:  # verify membership has needed columns
    raise NameError("Step 8 expects membership to contain columns: ['bucket','R_next'].")  # stop early if missing

MIN_SIGMA_OBS = 8                                     # require at least 8 non-missing observations in the rolling window to compute volatility
EPS = 1e-8                                            # tiny constant to prevent division-by-zero when computing 1/sigma
CAP_LO, CAP_HI = -0.95, 5.0                           # cap (winsorize) next-month returns to reduce outlier influence

Q_val = pd.to_numeric(membership["bucket"], errors="coerce").max()  # infer number of buckets (Q) from max bucket label
if pd.isna(Q_val):                                    # if max is NaN, then bucket column is all missing or non-numeric
    raise ValueError("Cannot infer Q because membership['bucket'] is all missing.")  # stop because we cannot compute winner/loser buckets
Q = int(Q_val)                                        # convert bucket count to an integer (e.g., 5 for quintiles)

# ------------------------------------------------------------  # visual separator for readability
# 0) Compute sigma_rt from step5 using shift(2).rolling(11)       # compute historical vol per stock per month
# ------------------------------------------------------------  # visual separator for readability
tmp = step5[["month","permno","R_t1"]].copy()          # start from step5 and keep only what we need to compute historical volatility
tmp["month"]  = pd.PeriodIndex(tmp["month"].astype(str), freq="M")  # normalize month to monthly PeriodIndex for correct sorting/grouping
tmp["permno"] = pd.to_numeric(tmp["permno"], errors="coerce").astype("Int64")  # coerce permno IDs to integer (nullable Int64)
tmp["R_t1"]   = pd.to_numeric(tmp["R_t1"], errors="coerce")         # coerce gross returns to numeric; invalid parsing becomes NaN
tmp = tmp.sort_values(["permno","month"]).reset_index(drop=True)    # sort by permno then month before rolling operations

tmp["sigma_rt"] = (                                   # create sigma_rt column to store rolling volatility at each (permno, month)
    tmp.groupby("permno", group_keys=False)["R_t1"]    # within each permno, take the R_t1 series in chronological order
       .apply(lambda s: s.shift(2).rolling(window=11, min_periods=MIN_SIGMA_OBS).std(ddof=1))  # std of (t-12..t-2), skip t-1, no look-ahead
)                                                     # close sigma_rt assignment

sigma_table = tmp[["permno","month","sigma_rt"]].dropna(subset=["sigma_rt"]).copy()  # keep only permno-month rows where sigma_rt was computed
sigma_table = sigma_table[sigma_table["sigma_rt"] > 0].copy()  # remove non-positive sigmas (cannot invert and meaningless for vol)

# ------------------------------------------------------------  # visual separator for readability
# (ADDED) Step-9-style table: sigma distribution                 # summarize sigma_rt across all rows
# ------------------------------------------------------------  # visual separator for readability
print("\nStep 8 check (sigma_rt distribution):")       # print a header label for the sigma diagnostic table
display(sigma_table[["sigma_rt"]].describe())          # show count/mean/std/min/quantiles/max of sigma_rt across all rows

# ------------------------------------------------------------  # visual separator for readability
# 1) Start from Step 6 membership, cap R_next                     # prep membership rows for weighting
# ------------------------------------------------------------  # visual separator for readability
p = membership.copy()                                  # create working copy of membership so original remains unchanged
p["month"]  = pd.PeriodIndex(p["month"].astype(str), freq="M")  # ensure month is monthly PeriodIndex (consistent join key)
p["permno"] = pd.to_numeric(p["permno"], errors="coerce").astype("Int64")  # ensure permno is Int64 for consistent join key
p["bucket"] = pd.to_numeric(p["bucket"], errors="coerce")       # ensure bucket labels are numeric (will later cast to int64)
p["R_next"] = pd.to_numeric(p["R_next"], errors="coerce")       # ensure realized next-month return is numeric
p["R_next_cap"] = p["R_next"].clip(lower=CAP_LO, upper=CAP_HI)  # cap/winsorize extreme next-month returns to stabilize averages

# ------------------------------------------------------------  # visual separator for readability
# 2) Attach sigma_rt at month t (no look-ahead)                   # join vol estimates onto membership
# ------------------------------------------------------------  # visual separator for readability
p = p.merge(sigma_table, on=["permno","month"], how="left")  # attach sigma_rt to each membership row using (permno, month) keys
p = p.dropna(subset=["bucket","R_next_cap","sigma_rt"]).copy()  # require bucket assignment, capped next return, and sigma to proceed
p["bucket"] = p["bucket"].astype("int64")                 # cast bucket to integer for pivot column indexing and comparisons

# ------------------------------------------------------------  # visual separator for readability
# 3) Inverse-vol weights within each (month, bucket)              # compute IV weights and normalize inside each bucket each month
# ------------------------------------------------------------  # visual separator for readability
p["w_raw"]  = 1.0 / (p["sigma_rt"] + EPS)                 # raw inverse-vol weight: higher weight for lower volatility names
p["w_norm"] = p["w_raw"] / p.groupby(["month","bucket"])["w_raw"].transform("sum")  # normalize weights to sum to 1 within each (month, bucket)

# ------------------------------------------------------------  # visual separator for readability
# (ADDED) Step-9-style table: position-level inputs used in IV weighting  # inspect mechanics inputs
# ------------------------------------------------------------  # visual separator for readability
print("\nStep 8 check (position-level inputs used):")     # print a header label for the position-level diagnostics table
cols_chk = ["sigma_rt", "w_raw", "w_norm", "R_next", "R_next_cap"]  # define the columns to summarize for IV weighting mechanics
display(p[cols_chk].apply(pd.to_numeric, errors="coerce").describe())  # display describe() for the selected inputs (across all membership rows)

# ------------------------------------------------------------  # visual separator for readability
# 4) Compute bucket return at t+1 using inverse-vol weights        # produce month×bucket returns under IV weighting
# ------------------------------------------------------------  # visual separator for readability
bucket_iv = (                                             # start building a table of bucket returns under IV weighting
    p.groupby(["month","bucket"])                          # group by signal month and bucket label
     .apply(lambda df: float(np.sum(df["w_norm"].values * df["R_next_cap"].values)))  # compute weighted average next-month return
     .rename("ret_iv")                                     # name the resulting Series as ret_iv
     .reset_index()                                        # convert group keys back into columns month/bucket
)                                                         # close bucket_iv assignment

wide_iv = bucket_iv.pivot(index="month", columns="bucket", values="ret_iv").sort_index()  # pivot to wide form: rows=month, cols=bucket, vals=ret_iv

LS_spread_iv = (wide_iv[Q] - wide_iv[1]).rename("LS_spread_iv")  # long-short spread using winner (Q) minus loser (1) bucket returns
LS_5050_iv   = (0.5 * (wide_iv[Q] - wide_iv[1])).rename("LS_5050_iv")  # 50/50 self-financing return: half the spread

# ------------------------------------------------------------  # visual separator for readability
# (ADDED) Step-9-style table: month-level winner/loser + LS        # summarize month-level bucket returns and LS
# ------------------------------------------------------------  # visual separator for readability
step8_check_month = pd.DataFrame({                       # create a month-level table with winner/loser returns and LS series
    "loser_bucket_ret_iv":  pd.to_numeric(wide_iv.get(1), errors="coerce"),  # loser bucket (1) IV return per month (may have NaNs)
    "winner_bucket_ret_iv": pd.to_numeric(wide_iv.get(Q), errors="coerce"),  # winner bucket (Q) IV return per month (may have NaNs)
    "LS_spread_iv":         pd.to_numeric(LS_spread_iv, errors="coerce"),    # LS spread series per month
    "LS_5050_iv":           pd.to_numeric(LS_5050_iv, errors="coerce"),      # 50/50 LS series per month
})                                                       # close DataFrame construction
print("\nStep 8 check (month-level IV winner/loser + LS):")  # print a header label for month-level diagnostic table
display(step8_check_month.describe())                      # show describe() for those month-level series (count, mean, std, etc.)

print("\nLS_5050_iv months:", int(pd.to_numeric(LS_5050_iv, errors="coerce").dropna().shape[0]))  # print number of non-NaN months in LS_5050_iv
display(pd.to_numeric(LS_5050_iv, errors="coerce").dropna().head(10))  # show first 10 non-NaN monthly returns of LS_5050_iv

# ------------------------------------------------------------  # visual separator for readability
# 5) Performance summary (same style as Step 7)                     # compute standard performance metrics
# ------------------------------------------------------------  # visual separator for readability
def eval_series_step8(R: pd.Series, name: str) -> dict:   # define a helper to compute mean/vol/sharpe/tstat and annualized metrics
    R = pd.to_numeric(R, errors="coerce").dropna().copy()  # coerce to numeric, drop NaNs, and work on a copy
    T = int(R.shape[0])                                   # number of observations (months)

    mean_m = float(R.mean())                              # average monthly return
    vol_m  = float(R.std(ddof=1))                         # sample std of monthly return
    sharpe_m = mean_m / vol_m if vol_m > 0 else np.nan    # monthly Sharpe ratio (rf assumed 0)
    tstat = mean_m / (vol_m / np.sqrt(T)) if (vol_m > 0 and T > 1) else np.nan  # t-stat of mean return

    ann_ret = 12.0 * mean_m                               # annualized mean return (linear scaling)
    ann_vol = np.sqrt(12.0) * vol_m                       # annualized volatility
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan  # annualized Sharpe ratio

    can_compound = bool(((1.0 + R) > 0).all())            # check if geometric compounding is mathematically valid

    return {                                              # pack the computed metrics into a dict for easy DataFrame construction
        "name": name,                                     # series label
        "T": T,                                           # number of months
        "mean_monthly": mean_m,                           # mean monthly return
        "vol_monthly": vol_m,                             # monthly volatility
        "sharpe_monthly": sharpe_m,                       # monthly Sharpe
        "tstat": tstat,                                   # t-stat of mean
        "ann_ret": ann_ret,                               # annualized return
        "ann_vol": ann_vol,                               # annualized vol
        "ann_sharpe": ann_sharpe,                         # annualized Sharpe
        "min": float(R.min()) if T else np.nan,           # minimum observed monthly return
        "max": float(R.max()) if T else np.nan,           # maximum observed monthly return
        "can_compound": can_compound,                     # whether compounded wealth curve is valid
    }                                                     # end dict

summary_iv = pd.DataFrame([                               # create a summary table with one row per IV series
    eval_series_step8(LS_5050_iv,  "LS_5050_iv (inv-vol, 50/50)"),  # evaluate IV 50/50 series
    eval_series_step8(LS_spread_iv,"LS_spread_iv (inv-vol, spread)") # evaluate IV spread series
])                                                       # close DataFrame construction

print("\n=== Step 8 Inverse-Vol Performance Summary ===")   # print header for performance summary
display(summary_iv)                                        # display the performance summary table

# ------------------------------------------------------------  # visual separator for readability
# 6) Plots: cumulative additive + cumulative compounded (if valid)  # show both additive and (when possible) compounded curves
# ------------------------------------------------------------  # visual separator for readability

# ---- 6A) LS_5050_iv additive cumulative                          # first: additive curve for LS_5050_iv
plt.figure()                                                     # start a new figure for LS_5050_iv additive cumulative plot
plt.plot(LS_5050_iv.index.to_timestamp(), pd.to_numeric(LS_5050_iv, errors="coerce").cumsum().values)  # plot additive cumulative sum
plt.title("LS_5050 inverse-vol cumulative (additive)")           # set plot title
plt.xlabel("Month")                                              # set x-axis label
plt.ylabel("Cumulative sum")                                     # set y-axis label
plt.show()                                                       # render plot

# ---- 6B) LS_5050_iv compounded cumulative (if valid)             # second: compounded curve for LS_5050_iv (only if defined)
R_5050 = pd.to_numeric(LS_5050_iv, errors="coerce").dropna().copy()  # make a clean numeric series for compounding check/plot
can_compound_5050 = bool(((1.0 + R_5050) > 0).all())              # compounding requires (1+R)>0 for every included month
if can_compound_5050:                                             # if compounding is mathematically valid
    cumprod_5050 = (1.0 + R_5050).cumprod() - 1.0                 # compute compounded cumulative return (wealth - 1)
    plt.figure()                                                  # start a new figure for compounded curve
    plt.plot(R_5050.index.to_timestamp(), cumprod_5050.values)    # plot compounded cumulative return over time
    plt.title("LS_5050 inverse-vol cumulative (compounded)")      # set plot title for compounded curve
    plt.xlabel("Month")                                           # set x-axis label
    plt.ylabel("Cumulative return")                               # set y-axis label (compounded return)
    plt.show()                                                    # render compounded plot
else:                                                             # if compounding is invalid (some month has R <= -1)
    print("WARNING: Cannot compound LS_5050_iv because some months have (1+R) <= 0.")  # print warning (same behavior as your Step 8 Strategy 2)

# ---- 6C) LS_spread_iv additive cumulative                         # first: additive curve for LS_spread_iv
plt.figure()                                                      # start a new figure for LS_spread_iv additive cumulative plot
plt.plot(LS_spread_iv.index.to_timestamp(), pd.to_numeric(LS_spread_iv, errors="coerce").cumsum().values)  # plot additive cumulative sum
plt.title("LS_spread inverse-vol cumulative (additive)")          # set plot title
plt.xlabel("Month")                                               # set x-axis label
plt.ylabel("Cumulative sum")                                      # set y-axis label
plt.show()                                                        # render plot

# ---- 6D) LS_spread_iv compounded cumulative (if valid)            # second: compounded curve for LS_spread_iv (only if defined)
R_spread = pd.to_numeric(LS_spread_iv, errors="coerce").dropna().copy()  # make a clean numeric series for compounding check/plot
can_compound_spread = bool(((1.0 + R_spread) > 0).all())           # compounding requires (1+R)>0 for every included month
if can_compound_spread:                                            # if compounding is mathematically valid
    cumprod_spread = (1.0 + R_spread).cumprod() - 1.0              # compute compounded cumulative return (wealth - 1)
    plt.figure()                                                   # start a new figure for compounded curve
    plt.plot(R_spread.index.to_timestamp(), cumprod_spread.values) # plot compounded cumulative return over time
    plt.title("LS_spread inverse-vol cumulative (compounded)")     # set plot title for compounded curve
    plt.xlabel("Month")                                            # set x-axis label
    plt.ylabel("Cumulative return")                                # set y-axis label (compounded return)
    plt.show()                                                     # render compounded plot
else:                                                              # if compounding is invalid (some month has R <= -1)
    print("WARNING: Cannot compound LS_spread_iv because some months have (1+R) <= 0.")  # print warning (same behavior as your Step 8 Strategy 2)

# ------------------------------------------------------------  # visual separator for readability
# 7) Histograms                                                     # visualize return distributions
# ------------------------------------------------------------  # visual separator for readability
plt.figure()                                                       # start a new figure for LS_5050_iv histogram
vals_5050 = pd.to_numeric(LS_5050_iv, errors="coerce").dropna().values  # extract non-NaN LS_5050_iv values as a NumPy array
plt.hist(vals_5050, bins=30)                                       # plot histogram with 30 bins
plt.title("Histogram — LS_5050_iv (inverse-vol, 50/50)")           # set histogram title
plt.xlabel("Monthly return")                                       # set x-axis label
plt.ylabel("Count")                                                # set y-axis label
plt.show()                                                         # render histogram

plt.figure()                                                       # start a new figure for LS_spread_iv histogram
vals_spread = pd.to_numeric(LS_spread_iv, errors="coerce").dropna().values  # extract non-NaN LS_spread_iv values as a NumPy array
plt.hist(vals_spread, bins=30)                                     # plot histogram with 30 bins
plt.title("Histogram — LS_spread_iv (inverse-vol, spread)")        # set histogram title
plt.xlabel("Monthly return")                                       # set x-axis label
plt.ylabel("Count")                                                # set y-axis label
plt.show()                                                         # render histogram

# ============================================================  # section header
# Step 9 (Strategy 1): Transaction costs & slippage (NET returns)  # what this step is
#                                                                  # spacer
# High-level pipeline:                                             # outline
#   (1) Position-level: build NET return R_t1_net from gross position table `all_monthly`  # stage 1
#       - adds entry_eff, exit_eff, roundtrip_cost, R_t1_net        # columns added
#       - (ADDED) position-level describe table (economics)         # diagnostics
#                                                                  # spacer
#   (2) Signal-level: recompute MOM_rt_net from R_t1_net (no look-ahead, skip t-1)  # stage 2
#                                                                  # spacer
#   (3) Portfolio-level: re-run bucket formation on MOM_rt_net and evaluate next-month net returns  # stage 3
#       - produces LS_spread_net, LS_5050_net                       # outputs
#       - (ADDED) month-level describe table: long_ret_next_net, short_ret_next_net, LS_spread_net, LS_5050_net  # diagnostics
#       - (ADDED) membership-level describe table: R_next_net, R_next_net_cap  # diagnostics
# ============================================================  # section footer

# -----------------------------  # separator
# 0) Parameters (edit these)     # parameters block
# -----------------------------  # separator
slippage_rate_entry = 0.001                              # 0.10% entry slippage: buy pays ABOVE mid
exit_cost_rate      = 0.001                              # 0.10% exit haircut: receive BELOW intrinsic
cost_bps_per_leg    = 5                                  # 5 bps per leg round-trip cost
n_legs              = 2                                  # straddle has 2 legs: call + put
cost_rate_total     = (cost_bps_per_leg / 10000.0) * n_legs  # bps -> decimal, multiply by legs

CAP_LO, CAP_HI = -0.95, 5.0                              # cap extreme option returns (winsorize)
Q = 5                                                    # number of momentum buckets (quintiles)

# -----------------------------  # separator
# 1) Position-level: build NET returns  # build net position returns
# -----------------------------  # separator
all_monthly_9 = all_monthly.copy()                       # working copy (do NOT overwrite gross table)

# Ensure numeric types (prevents object/string issues)     # type safety comment
for c in ["entry_mid", "exit_intrinsic", "R_t1"]:         # key columns needed to compute net return
    if c in all_monthly_9.columns:                       # guard if column missing
        all_monthly_9[c] = pd.to_numeric(all_monthly_9[c], errors="coerce")  # non-numeric -> NaN

# Validity mask for computing costs/returns                # validity definition comment
valid = (                                                 # boolean mask for “good” rows
    all_monthly_9["entry_mid"].notna() & (all_monthly_9["entry_mid"] > 0) &          # entry premium must be > 0
    all_monthly_9["exit_intrinsic"].notna() & (all_monthly_9["exit_intrinsic"] >= 0) # intrinsic payoff is >= 0
)                                                         # end validity mask

# Effective entry execution price (pay worse than mid when buying)  # entry execution modeling
all_monthly_9["entry_eff"] = np.where(                    # compute effective entry price with slippage
    valid,                                                  # compute only for valid rows
    all_monthly_9["entry_mid"] * (1.0 + slippage_rate_entry),# entry mid inflated by slippage
    np.nan                                                  # invalid rows -> NaN
)                                                         # end entry_eff assignment

# Effective exit execution value (receive worse than intrinsic)  # exit execution modeling
all_monthly_9["exit_eff"] = np.where(                     # compute effective exit value with haircut
    valid,                                                  # compute only for valid rows
    all_monthly_9["exit_intrinsic"] * (1.0 - exit_cost_rate),# intrinsic haircut
    np.nan                                                  # invalid rows -> NaN
)                                                         # end exit_eff assignment

# Round-trip cost modeled as % of entry premium notional   # cost model
all_monthly_9["roundtrip_cost"] = np.where(               # compute round-trip cost
    valid,                                                  # compute only for valid rows
    all_monthly_9["entry_mid"] * cost_rate_total,            # proportional cost on entry premium
    np.nan                                                  # invalid rows -> NaN
)                                                         # end roundtrip_cost assignment

# Net return (denominator uses entry_mid for comparability with your original R_t1 definition)  # return definition
all_monthly_9["R_t1_net"] = np.where(                     # compute net return
    valid,                                                  # compute only for valid rows
    (all_monthly_9["exit_eff"] - all_monthly_9["entry_eff"] - all_monthly_9["roundtrip_cost"])
    / all_monthly_9["entry_mid"],                           # normalize by entry_mid
    np.nan                                                  # invalid rows -> NaN
)                                                         # end R_t1_net assignment

# -----------------------------  # separator
# (ADDED) Step 9 describe tables — Position-level economics  # diagnostics: position-level
# -----------------------------  # separator
print("\nStep 9 check (gross vs net subset):")             # quick “gross vs net” diagnostic
display(all_monthly_9[["entry_mid", "exit_intrinsic", "R_t1", "R_t1_net"]].describe())  # summary stats for subset

print("\nStep 9 check (position-level economics, expanded):")  # expanded economics table header
pos_cols = ["entry_mid", "exit_intrinsic", "R_t1", "entry_eff", "exit_eff", "roundtrip_cost", "R_t1_net"]  # columns to summarize
pos_cols = [c for c in pos_cols if c in all_monthly_9.columns]  # guard if any missing
display(all_monthly_9[pos_cols].describe())                 # describe() at the position/trade level

# -----------------------------  # separator
# 2) Recompute Step 5 MOM using NET returns (no look-ahead)  # recompute momentum on net returns
# -----------------------------  # separator
step5_net = all_monthly_9.copy()                            # start from net position-level table

step5_net["permno"] = pd.to_numeric(step5_net["permno"], errors="coerce").astype("Int64")  # standardize permno

# Make secid safe (do NOT use .get(...).astype(...) because that can break if column missing)  # defensive programming
if "secid" in step5_net.columns:                            # if secid exists, coerce to Int64
    step5_net["secid"] = pd.to_numeric(step5_net["secid"], errors="coerce").astype("Int64")  # clean secid
else:                                                       # if secid missing, create an all-missing column
    step5_net["secid"] = pd.Series(pd.NA, index=step5_net.index, dtype="Int64")  # placeholder secid

step5_net["R_t1_net"] = pd.to_numeric(step5_net["R_t1_net"], errors="coerce")  # ensure net returns numeric

# Build a sortable time index for rolling (month string -> monthly Period -> Timestamp)  # time index for rolling
step5_net["_month_ts"] = pd.PeriodIndex(step5_net["month"].astype(str), freq="M").to_timestamp()  # month to timestamp

# Sort within each permno before rolling               # required order for rolling stats
step5_net = step5_net.sort_values(["permno", "_month_ts"]).reset_index(drop=True)  # sort for shift/rolling

# MOM_rt_net(t) = mean(R_net(t-12) ... R_net(t-2)) using shift(2).rolling(11)  # momentum definition
step5_net["MOM_rt_net"] = (                               # compute momentum signal
    step5_net.groupby("permno", group_keys=False)["R_t1_net"]  # per permno series
            .apply(lambda s: s.shift(2).rolling(window=11, min_periods=8).mean())  # no look-ahead (skip t-1)
)                                                         # end MOM_rt_net assignment

# -----------------------------  # separator
# 3) Step 6 on NET: bucketize on MOM_rt_net, evaluate next-month net returns  # net portfolio formation/eval
# -----------------------------  # separator
panel6_net = step5_net[["month", "permno", "R_t1_net", "MOM_rt_net"]].copy()  # keep only needed columns

panel6_net["permno"] = pd.to_numeric(panel6_net["permno"], errors="coerce").astype("Int64")  # permno clean
panel6_net["R_t1_net"] = pd.to_numeric(panel6_net["R_t1_net"], errors="coerce")              # net return clean
panel6_net["MOM_rt_net"] = pd.to_numeric(panel6_net["MOM_rt_net"], errors="coerce")          # momentum clean
panel6_net["month"] = pd.PeriodIndex(panel6_net["month"].astype(str), freq="M")              # month as PeriodIndex

# Next-month realized net return per stock (R_next_net at signal month t is realized in t+1)  # next-month return
panel6_net = panel6_net.sort_values(["permno", "month"]).reset_index(drop=True)              # required for shift
panel6_net["R_next_net"] = panel6_net.groupby("permno")["R_t1_net"].shift(-1)                # next-month return

# Keep only rows where we have both signal (MOM) and evaluation (R_next_net)  # required rows only
panel6_net = panel6_net.dropna(subset=["permno", "month", "MOM_rt_net", "R_next_net"]).copy()  # drop missing signal/eval

# Cap extreme net returns (tails can still dominate)  # winsorization
panel6_net["R_next_net_cap"] = panel6_net["R_next_net"].clip(lower=CAP_LO, upper=CAP_HI)     # cap net next returns

# Helper: robust bucket assignment within each month  # qcut helper
def bucketize_mom(s: pd.Series, q: int) -> pd.Series:        # s = MOM values in one month
    r = s.rank(method="first", ascending=True)               # rank breaks ties deterministically
    if r.size < q:                                           # cannot form q buckets if < q names
        return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")  # all NA buckets
    try:
        b = pd.qcut(r, q=q, labels=False, duplicates="drop") # 0..q-1 bins on ranked values
        if pd.Series(b).nunique(dropna=True) < q:            # if qcut collapses bins, abandon this month
            return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")  # all NA buckets
        return pd.Series((b + 1).astype("int64"), index=s.index, dtype="Int64")  # 1..q labels
    except Exception:
        return pd.Series([pd.NA] * r.size, index=s.index, dtype="Int64")         # fallback: NA buckets

# Assign bucket labels within each month using MOM_rt_net  # bucket assignment call
panel6_net["bucket"] = panel6_net.groupby("month")["MOM_rt_net"].transform(lambda s: bucketize_mom(s, Q))  # bucketize

# Drop rows/months where bucket assignment failed  # drop NA bucket rows
panel6_net = panel6_net.dropna(subset=["bucket"]).copy()     # keep only rows with bucket assigned

# Ensure bucket is int for pivot indexing  # type fix
panel6_net["bucket"] = panel6_net["bucket"].astype("int64")  # bucket as int

# Equal-weight bucket returns: mean next-month capped net return inside each bucket  # compute bucket returns
group_returns_net = (                                        # long-form bucket returns
    panel6_net.groupby(["month", "bucket"])                  # group by month and bucket
             .agg(n=("permno", "size"),                      # count of names in bucket
                  group_ret_next=("R_next_net_cap", "mean")) # equal-weight mean return
             .reset_index()                                  # back to columns
             .sort_values(["month", "bucket"])               # sort for readability
)                                                           # end group_returns_net

# Wide table: index=month, columns=bucket number, values=bucket return  # pivot to wide
wide_net = group_returns_net.pivot(index="month", columns="bucket", values="group_ret_next").sort_index()  # wide panel

# Winner/loser legs (used for “long_ret_next_net / short_ret_next_net” describe table)  # define legs
long_ret_next_net  = wide_net[Q] if Q in wide_net.columns else pd.Series(index=wide_net.index, dtype="float64")  # winner leg
short_ret_next_net = wide_net[1] if 1 in wide_net.columns else pd.Series(index=wide_net.index, dtype="float64")  # loser leg

# Long-short series  # define LS
LS_spread_net = (long_ret_next_net - short_ret_next_net).rename("LS_spread_net")            # spread (winner-loser)
LS_5050_net   = (0.5 * (long_ret_next_net - short_ret_next_net)).rename("LS_5050_net")      # 50/50 self-financing

# -----------------------------  # separator
# (ADDED) Step 9 describe tables — Month-level + Membership-level (net)  # diagnostics: month+membership
# -----------------------------  # separator
print("\nStep 9 check (month-level inputs, net):")          # header for month-level table
step9_month_inputs_net = pd.DataFrame({                     # build month-level diagnostics DataFrame
    "long_ret_next_net":  pd.to_numeric(long_ret_next_net,  errors="coerce"),  # coerce winner leg
    "short_ret_next_net": pd.to_numeric(short_ret_next_net, errors="coerce"),  # coerce loser leg
    "LS_spread_net":      pd.to_numeric(LS_spread_net,      errors="coerce"),  # coerce LS spread
    "LS_5050_net":        pd.to_numeric(LS_5050_net,        errors="coerce"),  # coerce LS 50/50
})                                                           # end DataFrame
display(step9_month_inputs_net.describe())                  # describe() of month-level legs + LS series

print("\nStep 9 check (membership-level, net):")            # header for membership-level table
mem9_net = panel6_net[["R_next_net", "R_next_net_cap"]].copy()  # membership-level realized returns
mem9_net["R_next_net"] = pd.to_numeric(mem9_net["R_next_net"], errors="coerce")              # ensure numeric
mem9_net["R_next_net_cap"] = pd.to_numeric(mem9_net["R_next_net_cap"], errors="coerce")      # ensure numeric
display(mem9_net.describe())                                 # describe() across all (permno, month) rows

# -----------------------------  # separator
# 4) Step 7-style evaluation (net) + plots + histograms  # evaluation and visuals
# -----------------------------  # separator
def eval_series(R: pd.Series, name: str) -> dict:           # evaluate a monthly series
    R = pd.to_numeric(R, errors="coerce").dropna().copy()   # numeric + drop NaN
    T = int(R.shape[0])                                     # number of months
    mean_m = float(R.mean())                                # mean monthly return
    vol_m  = float(R.std(ddof=1))                           # sample std
    sharpe_m = mean_m / vol_m if vol_m > 0 else np.nan      # monthly Sharpe
    tstat = mean_m / (vol_m / np.sqrt(T)) if (vol_m > 0 and T > 1) else np.nan  # t-stat

    ann_ret = 12.0 * mean_m                                 # annualized mean (linear)
    ann_vol = np.sqrt(12.0) * vol_m                         # annualized vol
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan  # annualized Sharpe

    can_compound = bool(((1.0 + R) > 0).all())              # compound valid if all (1+R)>0
    return {                                                 # return dict of metrics
        "name": name,                                       # series name
        "T": T,                                             # number of months
        "mean_monthly": mean_m,                             # monthly mean
        "vol_monthly": vol_m,                               # monthly std
        "sharpe_monthly": sharpe_m,                         # monthly Sharpe
        "tstat": tstat,                                     # t-stat of mean
        "ann_ret": ann_ret,                                 # annualized return
        "ann_vol": ann_vol,                                 # annualized vol
        "ann_sharpe": ann_sharpe,                           # annualized Sharpe
        "min": float(R.min()) if T else np.nan,             # min month
        "max": float(R.max()) if T else np.nan,             # max month
        "can_compound": can_compound,                       # compound feasibility flag
    }                                                       # end dict

# Summary table (net)  # performance summary build
summary_net = pd.DataFrame([                                # create summary DataFrame
    eval_series(LS_5050_net,  "LS_5050_net (50/50, net)"),   # evaluate 50/50 net series
    eval_series(LS_spread_net,"LS_spread_net (spread, net)") # evaluate spread net series
])                                                          # end DataFrame

print("\n=== Step 9 Net Performance Summary ===")            # print header
display(summary_net)                                        # display summary table

# Additive cumulative plots (PnL-like cumulative sum)  # additive plots section
plt.figure()                                                # new figure
plt.plot(LS_5050_net.index.to_timestamp(), LS_5050_net.cumsum().values)  # line plot (additive cumsum)
plt.title(f"Net LS_5050 cumulative (additive) (Q={Q})")      # title
plt.xlabel("Month")                                         # x label
plt.ylabel("Cumulative sum")                                # y label
plt.show()                                                  # render

plt.figure()                                                # new figure
plt.plot(LS_spread_net.index.to_timestamp(), LS_spread_net.cumsum().values)  # line plot (additive cumsum)
plt.title(f"Net LS_spread cumulative (additive) (Q={Q})")    # title
plt.xlabel("Month")                                         # x label
plt.ylabel("Cumulative sum")                                # y label
plt.show()                                                  # render

# (ADDED) Compounded cumulative plots (wealth-like cumprod), with warnings if invalid  # compounded plots section

R_5050_net = pd.to_numeric(LS_5050_net, errors="coerce").dropna().copy()     # clean numeric LS_5050_net for compounding
can_compound_5050 = bool(((1.0 + R_5050_net) > 0).all())                     # compounding requires (1+R)>0 for every month
if can_compound_5050:                                                       # if compounding is valid
    cumprod_5050_net = (1.0 + R_5050_net).cumprod() - 1.0                   # compounded cumulative return (wealth - 1)
    plt.figure()                                                            # new figure
    plt.plot(R_5050_net.index.to_timestamp(), cumprod_5050_net.values)      # plot compounded curve
    plt.title(f"Net LS_5050 cumulative (compounded) (Q={Q})")               # title
    plt.xlabel("Month")                                                    # x label
    plt.ylabel("Cumulative return")                                        # y label
    plt.show()                                                             # render
else:                                                                       # if compounding is NOT valid
    print("WARNING: Cannot compound LS_5050_net because some months have (1+R) <= 0.")  # warning like Strategy 2

R_spread_net = pd.to_numeric(LS_spread_net, errors="coerce").dropna().copy() # clean numeric LS_spread_net for compounding
can_compound_spread = bool(((1.0 + R_spread_net) > 0).all())                 # compounding requires (1+R)>0 for every month
if can_compound_spread:                                                     # if compounding is valid
    cumprod_spread_net = (1.0 + R_spread_net).cumprod() - 1.0               # compounded cumulative return (wealth - 1)
    plt.figure()                                                            # new figure
    plt.plot(R_spread_net.index.to_timestamp(), cumprod_spread_net.values)  # plot compounded curve
    plt.title(f"Net LS_spread cumulative (compounded) (Q={Q})")             # title
    plt.xlabel("Month")                                                    # x label
    plt.ylabel("Cumulative return")                                        # y label
    plt.show()                                                             # render
else:                                                                       # if compounding is NOT valid
    print("WARNING: Cannot compound LS_spread_net because some months have (1+R) <= 0.")  # warning like Strategy 2

# Histograms of monthly LS distributions (net)  # histogram section
plt.figure()                                                # new figure
plt.hist(pd.to_numeric(LS_5050_net, errors="coerce").dropna().values, bins=30)  # histogram
plt.title("Histogram — LS_5050_net (50/50, net)")            # title
plt.xlabel("Monthly return")                                # x label
plt.ylabel("Count")                                         # y label
plt.show()                                                  # render

plt.figure()                                                # new figure
plt.hist(pd.to_numeric(LS_spread_net, errors="coerce").dropna().values, bins=30)  # histogram
plt.title("Histogram — LS_spread_net (spread, net)")         # title
plt.xlabel("Monthly return")                                # x label
plt.ylabel("Count")                                         # y label
plt.show()                                                  # render
