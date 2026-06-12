import pandas as pd
from pathlib import Path
import numpy as np
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

# ============================================================  # section header: paths
# 0) Paths (edit only these)  # user-editable input paths
# ============================================================  # section header end
BASE_DIR   = Path(r"C:\Chris Academics\Icarus Fund Internship\Option Price Folder")  # base folder holding OptionMetrics daily parquet files
MAP_PATH   = Path(r"C:\Chris Academics\Icarus Fund Internship\us_stock_permno_with_secid.parquet")  # secid→permno mapping file (with validity windows)
STOCK_PATH = Path(r"C:\Chris Academics\Icarus Fund Internship\stock_price__1996_2022.parquet")  # CRSP-style stock price parquet (permno/date/prc)

# ============================================================  # section header: mapping load
# 1) Load secid -> permno mapping (validity windows)  # load link table used to map OptionMetrics secid to CRSP permno
# ============================================================  # section header end
link = pd.read_parquet(MAP_PATH).copy()  # read mapping parquet and copy to avoid mutating original
link["sdate"] = pd.to_datetime(link["sdate"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))  # start validity date (default very early)
link["edate"] = pd.to_datetime(link["edate"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))  # end validity date (default very late)
link["secid"] = pd.to_numeric(link["secid"], errors="coerce").astype("Int64")  # ensure secid is numeric nullable int

# ============================================================  # section header: OptionMetrics helpers
# 2) OptionMetrics helpers  # helper functions to locate/load daily option files
# ============================================================  # section header end
def opt_path(d: pd.Timestamp) -> Path:  # build file path for one OptionMetrics trading date
    return BASE_DIR / f"{d.year}" / f"optionmetrics_{d.strftime('%Y-%m-%d')}_nonzeroOI.parquet"  # year-subfolder + date-stamped parquet name

def load_opt_day(d: pd.Timestamp) -> pd.DataFrame:  # load one trading day of options and compute derived fields
    df = pd.read_parquet(opt_path(d))  # read that day's option parquet
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # coerce option quote date
    df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")  # coerce option expiration date
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0  # compute midquote
    df["spread"] = df["best_offer"] - df["best_bid"]  # compute bid-ask spread
    df["K"] = df["strike_price"] / 1000.0  # OptionMetrics strike scaled by 1000
    return df  # return enriched option dataframe

# ============================================================  # section header: stock loader
# 3) Stock loader (CRSP-style prc can be negative)  # helper to load stock prices and compute S=abs(prc)
# ============================================================  # section header end
def load_stock(stock_path: Path) -> pd.DataFrame:  # load CRSP-style stock data
    stock = pd.read_parquet(stock_path).copy()  # read parquet and copy
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")  # coerce trading date
    stock["permno"] = pd.to_numeric(stock["permno"], errors="coerce").astype("Int64")  # ensure permno is numeric nullable int
    stock["prc"] = pd.to_numeric(stock["prc"], errors="coerce")  # ensure prc is numeric
    # Use abs(prc) as price level (CRSP convention)  # CRSP uses negative prc as a flag; abs(prc) is price level
    stock["S"] = stock["prc"].abs()  # define stock price level S
    return stock  # return stock dataframe with S

# ============================================================  # section header: Step 4 helper
# 4) Step 4 helper: attach S_{t'} at (or before) E_t1  # merge stock price onto option positions at expiry date
# ============================================================  # section header end
def attach_stock_price_at_expiry(selected: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:  # attach stock price at/before expiry
    out = selected.copy()  # copy selected positions so we don't mutate caller data

    # types  # enforce key dtypes for merge_asof
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")  # ensure permno is numeric nullable int
    out["E_t1"] = pd.to_datetime(out["E_t1"], errors="coerce")  # ensure expiry date is datetime

    stock_small = stock[["permno", "date", "S"]].dropna(subset=["permno", "date", "S"]).copy()  # keep only needed columns + drop missing
    stock_small["permno"] = pd.to_numeric(stock_small["permno"], errors="coerce").astype("Int64")  # enforce permno dtype
    stock_small["date"] = pd.to_datetime(stock_small["date"], errors="coerce")  # enforce stock date dtype
    stock_small["S"] = pd.to_numeric(stock_small["S"], errors="coerce")  # enforce S numeric dtype

    # rename ON key  # merge_asof uses a single "on" time key (here: stock_date)
    stock_small = stock_small.rename(columns={"date": "stock_date"})  # rename date→stock_date for clarity and merge_asof

    # IMPORTANT: merge_asof requires the "on" keys sorted (globally), not just within permno  # required by pandas merge_asof
    stock_small = stock_small.sort_values(["stock_date", "permno"]).reset_index(drop=True)  # sort right table by time then permno
    out = out.sort_values(["E_t1", "permno"]).reset_index(drop=True)  # sort left table by time then permno

    out = pd.merge_asof(  # asof-merge to get the last stock observation at/before E_t1 per permno
        out,  # left table: option positions with E_t1
        stock_small,  # right table: stock prices with stock_date
        left_on="E_t1",  # left time key
        right_on="stock_date",  # right time key
        by="permno",  # match within permno
        direction="backward",  # take the latest stock_date <= E_t1
        allow_exact_matches=True,  # allow stock_date == E_t1
    )  # end merge_asof call
    return out  # return positions augmented with stock_date and S

# ============================================================  # section header: monthly straddle builder
# 5) (Steps 1–3) Build selected straddles for one month  # build straddles and compute intrinsic returns
#     + (Step 4) intrinsic payoff using STOCK price  # payoff uses stock S at expiry E_t1
# ============================================================  # section header end
def month_return_top20first(month: str, cal: pd.DataFrame, stock: pd.DataFrame) -> pd.DataFrame:  # build Top20 liquidity universe then best straddle per secid
    # calendar lookup (uses your existing cal variable/table)  # fetch formation and expiry dates for this month label
    F_t  = cal.loc[cal["month"] == month, "F_t"].iloc[0]  # formation date for month
    E_t1 = cal.loc[cal["month"] == month, "E_t1"].iloc[0]  # next-month expiry date for month

    # --- formation day options ---  # load options on formation date
    opt = load_opt_day(F_t)  # load OptionMetrics for F_t (adds mid/spread/K)

    std = opt["expiry_indicator"].isna() | (opt["expiry_indicator"].astype(str).str.strip() == "")  # identify standard (non-special) contracts
    opt_F1 = opt[std & (opt["exdate"] == E_t1)].copy()  # keep only standard options expiring at E_t1

    opt_F1 = opt_F1.dropna(  # drop rows missing any required pricing/greek/id fields
        subset=["best_bid","best_offer","mid","spread","delta","open_interest","K","secid","cp_flag","optionid"]  # required columns
    )  # end dropna
    opt_F1 = opt_F1[  # basic quote sanity filters
        (opt_F1["best_offer"] >= opt_F1["best_bid"]) &  # ask >= bid
        (opt_F1["best_offer"] > 0) &  # positive ask
        (opt_F1["best_bid"] > 0) &  # positive bid
        (opt_F1["mid"] > 0)  # positive mid
    ].copy()  # materialize filtered data

    opt_F1["secid"] = pd.to_numeric(opt_F1["secid"], errors="coerce").astype("Int64")  # ensure secid is numeric nullable int
    opt_F1 = opt_F1.dropna(subset=["secid"]).copy()  # drop rows where secid is missing after coercion

    # ---------- Top-20 secids FIRST (must have valid permno on F_t) ----------  # universe selection stage
    liq = (  # compute per-secid liquidity aggregates
        opt_F1.groupby("secid")  # group by underlying secid
        .agg(total_oi=("open_interest", "sum"),  # total open interest across contracts
             med_spread=("spread", "median"))  # median spread across contracts
        .reset_index()  # bring secid back as a column
    )  # end aggregation
    liq = liq.dropna(subset=["med_spread"]).copy()  # require median spread to be non-missing
    liq["liq_score"] = liq["total_oi"] / (1.0 + liq["med_spread"])  # liquidity score used for ranking (OI / (1+spread))

    cand = liq.merge(link[["secid","permno","sdate","edate"]], on="secid", how="left")  # attach mapping windows to secids
    cand = cand[(F_t >= cand["sdate"]) & (F_t <= cand["edate"])].copy()  # keep only mappings valid on formation date F_t

    # if multiple mapping windows, keep one row per secid (most recent sdate, but still valid)  # deduplicate secid mappings
    cand = (  # choose best mapping row per secid after sorting
        cand.sort_values(["liq_score","sdate"], ascending=[False, False])  # prefer higher liquidity score and more recent sdate
            .groupby("secid", as_index=False)  # group by secid
            .head(1) #keep the first row within each secid that has the greatest liquidity
    )  # end dedupe
    cand = cand.dropna(subset=["permno"]).copy()  # require a valid permno mapping

    top20 = cand.sort_values("liq_score", ascending=False).head(20).copy()  # pick top 20 secids by liquidity score
    top20_map = top20[["secid","permno"]].copy()  # keep mapping for those top 20 secids
    top20_secids = set(top20_map["secid"].tolist())  # set of top20 secid values for filtering

    # only keep option quotes for top20 secids  # restrict option universe to selected underlyings
    opt_F1 = opt_F1[opt_F1["secid"].isin(top20_secids)].copy()  # filter to top20 secids

    # ---------- Build straddles only within top20 (Steps 2–3) ----------  # construct call/put pairs and choose best per secid
    calls = opt_F1[  # call candidates
        (opt_F1["cp_flag"] == "C") &  # calls only
        (opt_F1["open_interest"] > 0) &  # require nonzero OI
        (opt_F1["delta"].between(0.25, 0.75, inclusive="both"))  # filter to mid-delta calls
    ].copy()  # materialize calls

    puts = opt_F1[  # put candidates
        (opt_F1["cp_flag"] == "P") &  # puts only
        (opt_F1["open_interest"] > 0)  # require nonzero OI
    ].copy()  # materialize puts

    # keep symbol/root only from calls  # keep identifying fields once (calls carry symbol/root)
    calls = calls[[  # select call columns needed for pairing and valuation
        "secid","symbol","root","date","exdate","K","strike_price","optionid","delta","mid","spread","open_interest"  # fields for calls
    ]].rename(columns={  # rename call-side fields to call_* names
        "optionid":"call_optionid",  # call option identifier
        "delta":"call_delta",  # call delta
        "mid":"call_mid",  # call midquote
        "spread":"call_spread",  # call spread
        "open_interest":"call_oi",  # call open interest
    })  # end rename

    puts = puts[[  # select put columns needed for pairing and valuation
        "secid","exdate","K","strike_price","optionid","delta","mid","spread","open_interest"  # fields for puts
    ]].rename(columns={  # rename put-side fields to put_* names
        "optionid":"put_optionid",  # put option identifier
        "delta":"put_delta",  # put delta
        "mid":"put_mid",  # put midquote
        "spread":"put_spread",  # put spread
        "open_interest":"put_oi",  # put open interest
    })  # end rename

    pairs = calls.merge(puts, on=["secid","exdate","K","strike_price"], how="inner")  # pair calls and puts with same secid/exdate/strike
    pairs["combined_spread"] = pairs["call_spread"] + pairs["put_spread"]  # combined bid-ask spread of straddle

    # delta-neutral weights  # compute weights so straddle is delta-neutral at entry
    pairs = pairs[(pairs["call_delta"] - pairs["put_delta"]).abs() > 1e-8].copy()  # avoid divide-by-zero when deltas nearly cancel
    pairs["wC"] = (-pairs["put_delta"]) / (pairs["call_delta"] - pairs["put_delta"])  # call weight for delta neutrality
    pairs["wP"] = 1.0 - pairs["wC"]  # put weight implied by call weight
    pairs = pairs[  # enforce weights are feasible convex combination
        pairs["wC"].between(0, 1, inclusive="both") &  # call weight between 0 and 1
        pairs["wP"].between(0, 1, inclusive="both")  # put weight between 0 and 1
    ].copy()  # materialize filtered pairs

    pairs["entry_mid"] = pairs["wC"] * pairs["call_mid"] + pairs["wP"] * pairs["put_mid"]  # weighted entry cost (straddle mid)

    # best straddle per secid  # choose one straddle per underlying
    best = (  # sort by tightest spread then cheapest entry, take first per secid
        pairs.sort_values(["secid","combined_spread","entry_mid"], ascending=[True, True, True])  # ranking criteria
             .groupby("secid", as_index=False)  # group by secid
             .head(1)  # keep best row per secid
             .copy()  # materialize
    )  # end best selection

    best["month"] = month  # store month label
    best["F_t"] = F_t  # store formation date
    best["E_t1"] = E_t1  # store expiry date

    selected = best[[  # select final fields to carry forward
        "month","F_t","E_t1",  # calendar fields
        "secid","symbol","root",  # identifiers
        "K","strike_price",  # strike fields
        "call_optionid","put_optionid",  # option ids
        "call_delta","put_delta","wC","wP",  # deltas and weights
        "call_mid","put_mid","entry_mid",  # midquotes and entry
        "call_spread","put_spread","combined_spread",  # spreads
        "call_oi","put_oi"  # open interest
    ]].copy()  # materialize selected positions table

    # attach permno  # map underlying secid to CRSP permno
    selected = selected.merge(top20_map, on="secid", how="left")  # merge permno onto selected positions

    # =======================================================  # section header: payoff/return
    # Step 4 (correct): use STOCK price at E_t1 to compute intrinsic payoff  # compute payoff from S at expiry
    # =======================================================  # section header end
    selected = attach_stock_price_at_expiry(selected, stock)  # adds stock_date, S

    selected["S"] = pd.to_numeric(selected["S"], errors="coerce")  # enforce numeric stock price
    selected["K"] = pd.to_numeric(selected["K"], errors="coerce")  # enforce numeric strike
    selected["entry_mid"] = pd.to_numeric(selected["entry_mid"], errors="coerce")  # enforce numeric entry cost

    selected["payoff_c"] = np.maximum(selected["S"] - selected["K"], 0.0)  # call intrinsic payoff at expiry
    selected["payoff_p"] = np.maximum(selected["K"] - selected["S"], 0.0)  # put intrinsic payoff at expiry

    selected["exit_intrinsic"] = selected["wC"] * selected["payoff_c"] + selected["wP"] * selected["payoff_p"]  # weighted straddle intrinsic value

    valid = (  # validity mask for computing returns
        selected["entry_mid"].notna() & (selected["entry_mid"] > 0) &  # positive entry cost
        selected["S"].notna() & (selected["S"] > 0) &  # positive stock price
        selected["K"].notna() & (selected["K"] > 0)  # positive strike
    )  # end validity mask

    selected["R_t1"] = np.where(  # compute holding-to-expiry return
        valid,  # only compute when valid
        (selected["exit_intrinsic"] - selected["entry_mid"]) / selected["entry_mid"],  # simple return
        np.nan  # otherwise missing
    )  # end np.where

    return selected[[  # return position-level outputs
        "month","F_t","E_t1","secid","permno","symbol","root",  # identifiers
        "K","entry_mid","stock_date","S",  # strike/entry/stock-at-expiry
        "payoff_c","payoff_p","exit_intrinsic","R_t1",  # payoff and return
        "call_optionid","put_optionid","wC","wP"         # option ids + weights (needed later for enhanced exits)
    ]].copy()  # return a clean copy

def load_mids_for_optionids(d: pd.Timestamp, optionids: set) -> pd.DataFrame:  # load mids for a subset of optionids on a given day
    """  # docstring start (explains function intent)
    Load one day's option file and return mid quotes for the optionids we care about.  # docstring content (not a Python comment)
    """  # docstring end
    opt = load_opt_day(d)  # your existing loader (adds mid)
    opt = opt.dropna(subset=["optionid", "mid"]).copy()  # require optionid and mid to be present
    opt["optionid"] = pd.to_numeric(opt["optionid"], errors="coerce")  # coerce optionid to numeric
    opt = opt.dropna(subset=["optionid"]).copy()  # drop rows where optionid could not be parsed
    opt["optionid"] = opt["optionid"].astype("int64")  # cast optionid to int64 for dictionary keys / isin

    opt = opt[opt["optionid"].isin(optionids)][["optionid", "mid"]].copy()  # filter to needed optionids and keep optionid+mid
    return opt  # return mids table for the requested optionids

def apply_combined_exits_one_month(  # apply stop-loss/profit-taking rules for one month’s positions
    sel_m: pd.DataFrame,  # selected positions for a single month (same F_t and E_t1)
    stop_alpha: float = 0.75,     # 75% drawdown => stop level = 0.25 * entry
    profit_beta: float = 0.80,    # +80% profit => profit level = 1.80 * entry
) -> pd.DataFrame:  # returns positions table with enhanced exits and R_t1_enh
    """  # docstring start (explains logic)
    For a given month (all rows share F_t and E_t1), scan daily mids and trigger:  # docstring content (not a Python comment)
      - STOP if combined <= (1-stop_alpha)*entry  # docstring content (not a Python comment)
      - PROFIT if combined >= (1+profit_beta)*entry  # docstring content (not a Python comment)
    If neither triggers, hold to expiry (use intrinsic as you already do).  # docstring content (not a Python comment)
    """  # docstring end
    out = sel_m.copy()  # copy input month positions

    # Sanity: required columns  # validate required inputs exist
    need = ["F_t","E_t1","call_optionid","put_optionid","wC","wP","entry_mid","K","permno"]  # columns needed to compute exits
    missing = [c for c in need if c not in out.columns]  # find missing required columns
    if missing:  # if any required columns missing
        raise ValueError(f"Missing required columns for exits: {missing}")  # stop with explicit error

    # Normalize types  # enforce dtypes for dates and numerics
    out["F_t"] = pd.to_datetime(out["F_t"])  # formation date as datetime
    out["E_t1"] = pd.to_datetime(out["E_t1"])  # expiry date as datetime
    out["entry_mid"] = pd.to_numeric(out["entry_mid"], errors="coerce")  # entry cost numeric
    out["wC"] = pd.to_numeric(out["wC"], errors="coerce")  # call weight numeric
    out["wP"] = pd.to_numeric(out["wP"], errors="coerce")  # put weight numeric

    out["call_optionid"] = pd.to_numeric(out["call_optionid"], errors="coerce")  # call optionid numeric
    out["put_optionid"] = pd.to_numeric(out["put_optionid"], errors="coerce")  # put optionid numeric

    # Exit bookkeeping  # initialize outputs for enhanced exit logic
    out["exit_reason"] = "HOLD"   # HOLD / STOP / PROFIT
    out["exit_date"] = pd.NaT  # the date when stop/profit triggers (NaT until triggered)
    out["exit_value"] = np.nan    # combined value at exit date (mid-based)
    out["R_t1_enh"] = np.nan  # enhanced realized return (stop/profit if triggered, else hold return)

    # Precompute trigger levels per row  # thresholds depend on entry_mid
    out["stop_level"] = (1.0 - stop_alpha) * out["entry_mid"]  # stop trigger level
    out["profit_level"] = (1.0 + profit_beta) * out["entry_mid"]  # profit trigger level

    # We only evaluate rows with clean entry  # do not process invalid entries
    active = out["entry_mid"].notna() & (out["entry_mid"] > 0)  # mask: valid entry_mid
    out.loc[~active, "exit_reason"] = "INVALID_ENTRY"  # tag invalid entries

    # Date range: start AFTER entry day  # begin scanning on business day after formation
    F_t = out["F_t"].iloc[0]  # formation date (shared for the month)
    E_t1 = out["E_t1"].iloc[0]  # expiry date (shared for the month)
    days = pd.date_range(F_t + pd.Timedelta(days=1), E_t1, freq="B")  # skips weekends

    # Set of optionids we need quotes for  # build quote universe needed for scanning
    # (only for currently active rows)  # only active rows contribute optionids
    active_rows = out[active].copy()  # subset to active entries
    call_ids = set(active_rows["call_optionid"].dropna().astype("int64").tolist())  # set of call optionids
    put_ids  = set(active_rows["put_optionid"].dropna().astype("int64").tolist())  # set of put optionids
    need_ids = call_ids | put_ids  # union: optionids needed to value combined straddle

    # Iterate daily; stop/profit is first-hit logic  # first time threshold hit is the exit
    still_open = set(active_rows.index.tolist())  # track indices that have not yet exited

    for d in days:  # loop through business days from after F_t through E_t1
        if not still_open:  # if all positions have exited
            break  # stop scanning early

        # Load mids for needed optionids on day d  # read daily midquotes for optionids
        try:  # attempt to load daily mids
            mids = load_mids_for_optionids(pd.Timestamp(d), need_ids)  # mids for needed optionids on date d
        except FileNotFoundError:  # if daily option file is missing
            continue  # skip that day

        if mids.empty:  # if no mids were returned
            continue  # nothing to evaluate that day

        mid_map = dict(zip(mids["optionid"].values, mids["mid"].values))  # map optionid -> mid for fast lookup

        # Evaluate triggers for still-open rows  # compute combined value for each open position
        idx_list = list(still_open)  # list of still-open indices (for loc/map)
        call_mid = out.loc[idx_list, "call_optionid"].map(lambda x: mid_map.get(int(x), np.nan) if pd.notna(x) else np.nan)  # lookup call mids
        put_mid  = out.loc[idx_list, "put_optionid"].map(lambda x: mid_map.get(int(x), np.nan) if pd.notna(x) else np.nan)  # lookup put mids

        # Need both legs to value combined  # combined is only valid if both call_mid and put_mid exist
        ok = call_mid.notna() & put_mid.notna()  # mask: both legs have quotes
        if not ok.any():  # if no rows have both legs quoted
            continue  # skip day

        combined = out.loc[idx_list, "wC"].values * call_mid.values + out.loc[idx_list, "wP"].values * put_mid.values  # combined straddle value

        # Apply triggers in time order (daily). If both happen same day (rare), prioritize STOP for safety.  # trigger priority rule
        stop_hit = ok.values & (combined <= out.loc[idx_list, "stop_level"].values)  # stop trigger condition
        prof_hit = ok.values & (combined >= out.loc[idx_list, "profit_level"].values) & (~stop_hit)  # profit trigger, excluding stop-hit rows

        hit_any = stop_hit | prof_hit  # rows that hit either trigger today
        if not hit_any.any():  # if no triggers today
            continue  # move to next day

        hit_idx = np.array(idx_list)[hit_any]  # indices that triggered today
        hit_val = combined[hit_any]  # combined values at trigger
        hit_reason = np.where(stop_hit[hit_any], "STOP", "PROFIT")  # label trigger type per hit index

        out.loc[hit_idx, "exit_reason"] = hit_reason  # record exit reason
        out.loc[hit_idx, "exit_date"] = pd.Timestamp(d)  # record exit date
        out.loc[hit_idx, "exit_value"] = hit_val  # record exit combined value

        # Realized return for early exits  # compute return using exit_value and entry_mid
        out.loc[hit_idx, "R_t1_enh"] = (out.loc[hit_idx, "exit_value"] - out.loc[hit_idx, "entry_mid"]) / out.loc[hit_idx, "entry_mid"]  # enhanced return for triggered exits

        # Remove from open set  # mark positions as closed after triggering
        for i in hit_idx:  # iterate triggered indices
            still_open.discard(i)  # remove from still-open set

    # For those still HOLD (no trigger), use your intrinsic-at-expiry return (already computed as R_t1)  # fill enhanced return for holds
    hold_mask = (out["exit_reason"] == "HOLD")  # rows that never triggered
    out.loc[hold_mask, "R_t1_enh"] = out.loc[hold_mask, "R_t1"]  # enhanced return equals hold-to-expiry return

    return out  # return enhanced month table with exit metadata and R_t1_enh

# ============================================================  # section header: run across months
# 6) Run across months  # loop over all months in calendar to build position table
#    IMPORTANT: this assumes you already have your calendar table  # prerequisite: cal exists
#    in a variable named `cal` with columns: ["F_t","E_t1","month"]  # required calendar columns
# ============================================================  # section header end
stock = load_stock(STOCK_PATH)  # load full stock price table once

results = []  # collect per-month selected position tables
for m in cal["month"]:  # iterate each month label in calendar
    try:  # attempt to compute that month’s positions and returns
        dfm = month_return_top20first(m, cal, stock)  # build monthly selected straddles + intrinsic return
        results.append(dfm)  # append monthly result
        print(m, "ok", "n=", len(dfm), "missing=", int(dfm["R_t1"].isna().sum()))  # log status and missing return count
    except FileNotFoundError as e:  # if option file for that month is missing
        print(m, "SKIP missing option file:", e)  # log skip reason

all_monthly = pd.concat(results, ignore_index=True)  # concatenate all months into one position-level table
print(all_monthly.head())  # preview first rows
print("shape:", all_monthly.shape)  # print table shape (rows, cols)

# below is added for the profit-booking and stop-loss  # enhanced exit logic (stop/profit scanning)
enhanced = []  # collect per-month enhanced tables
for m, df_m in all_monthly.groupby("month", sort=False):  # iterate by month (preserve original ordering)
    df_e = apply_combined_exits_one_month(df_m, stop_alpha=0.75, profit_beta=0.80)  # apply stop-loss/profit-taking rules
    enhanced.append(df_e)  # append enhanced month output

all_enh = pd.concat(enhanced, ignore_index=True)  # combine all enhanced months into one table

print(all_enh[["month","exit_reason","R_t1","R_t1_enh"]].head())  # preview key enhanced columns
print(all_enh["exit_reason"].value_counts(dropna=False))  # count how many holds/stops/profits/invalid
print(all_enh[["R_t1","R_t1_enh"]].describe())  # compare summary stats of base vs enhanced returns


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

# =========================  # section header: Step 6 description
# Step 6 (REVISED): Top-3 / Bottom-3 selection with fallback  # selects winner/loser groups by MOM with return-availability fallback
# Correct timing: sort on MOM_{t}, evaluate R_{t+1}  # ranks using momentum at time t and measures return in t+1
# Requires: step5 with columns ["month","permno","R_t1","MOM_rt"]  # input requirement (here using R_t1_enh instead of R_t1)
# Produces: LS_spread, LS_5050, membership (for Step 7 and reporting)  # outputs long-short series + membership table
# =========================  # section header end

N_LONG  = 3  # number of long positions (top momentum) each month
N_SHORT = 3  # number of short positions (bottom momentum) each month

panel6 = step5[["month","permno","R_t1_enh","MOM_rt"]].copy()  # build Step 6 panel using enhanced returns and momentum

# Standardize types  # ensure consistent dtypes before sorting/grouping
panel6["permno"] = pd.to_numeric(panel6["permno"], errors="coerce").astype("Int64")  # permno as nullable int key
panel6["R_t1_enh"] = pd.to_numeric(panel6["R_t1_enh"], errors="coerce")  # enhanced monthly return as numeric
panel6["MOM_rt"] = pd.to_numeric(panel6["MOM_rt"], errors="coerce")  # momentum signal as numeric
panel6["month"]  = pd.PeriodIndex(panel6["month"].astype(str), freq="M")  # month as Period[M] for clean monthly indexing

# 1) Build next-month return R_{t+1} within each permno  # construct forward return aligned with ranking month t
panel6 = panel6.sort_values(["permno","month"])  # sort within permno by month so shift(-1) is correct
panel6["R_next"] = panel6.groupby("permno")["R_t1_enh"].shift(-1)  # next-month return (t+1) per permno

# 2) Optional stabilization / winsorization for portfolio mechanics  # cap extreme next-month returns for robustness
# (Keep your original caps if you like)  # note: caps are adjustable
CAP_LO, CAP_HI = -0.95, 5.0  # lower/upper caps used to clip R_next
panel6["R_next_cap"] = panel6["R_next"].clip(lower=CAP_LO, upper=CAP_HI)  # clipped next-month return used in portfolio averages

# 3) We need MOM_t to rank. Keep MOM non-null.  # drop rows missing momentum since they cannot be ranked
panel6 = panel6.dropna(subset=["permno", "month", "MOM_rt"]).copy()  # keep only rows with valid permno/month/MOM_rt

def pick_top_bottom_with_fallback(df_m: pd.DataFrame, n_long: int, n_short: int):  # month-level selector with return-availability fallback
    """  # docstring start (explains selection logic)
    For a single month t:  # operate on a single month's cross-section
      - rank by MOM_rt (desc for long, asc for short)  # winners = high MOM, losers = low MOM
      - select first n_long names with valid R_next_cap  # ensure next-month return exists for long names
      - select first n_short names from the bottom with valid R_next_cap  # ensure next-month return exists for short names
    """  # docstring end
    # Sort for ranking  # prepare descending rank order for longs
    df_sorted = df_m.sort_values(["MOM_rt", "permno"], ascending=[False, True]).copy()  # sort by MOM desc, tie-break by permno

    # Long: go down the ranked list, keep only those with valid next-month return  # pick top names with non-missing R_next_cap
    longs = df_sorted[df_sorted["R_next_cap"].notna()].head(n_long).copy()  # take first n_long with valid next-month return

    # Short: go up from the bottom of the ranked list  # prepare ascending rank order for shorts
    df_sorted_asc = df_m.sort_values(["MOM_rt", "permno"], ascending=[True, True]).copy()  # sort by MOM asc, tie-break by permno
    shorts = df_sorted_asc[df_sorted_asc["R_next_cap"].notna()].head(n_short).copy()  # take first n_short from bottom with valid next-month return

    # If overlap occurs (possible in tiny universes), drop overlaps from shorts then refill  # prevent same permno in both sides
    if len(longs) > 0 and len(shorts) > 0:  # only check overlap if both sides non-empty
        overlap = set(longs["permno"].tolist()) & set(shorts["permno"].tolist())  # intersection of permnos in long and short sets
        if overlap:  # if any overlap exists
            shorts = shorts[~shorts["permno"].isin(overlap)].copy()  # drop overlapping permnos from shorts
            # refill shorts if needed  # rebuild shorts set using remaining candidates (excluding longs)
            refill = (  # construct refill candidate set
                df_sorted_asc[df_sorted_asc["R_next_cap"].notna() & ~df_sorted_asc["permno"].isin(set(longs["permno"])) ]  # valid returns and not in longs
                .head(n_short)  # take first n_short candidates
                .copy()  # materialize
            )  # end refill construction
            shorts = refill  # replace shorts with refilled set

    long_ret  = float(longs["R_next_cap"].mean())  if len(longs)  > 0 else np.nan  # average capped next-month return of long bucket
    short_ret = float(shorts["R_next_cap"].mean()) if len(shorts) > 0 else np.nan  # average capped next-month return of short bucket

    # membership rows  # create membership table with side labels for later reporting
    mem_long = longs.assign(side="LONG")  # tag long rows
    mem_short = shorts.assign(side="SHORT")  # tag short rows

    # return both  # return bucket means plus membership rows
    return long_ret, short_ret, pd.concat([mem_long, mem_short], ignore_index=True)  # combine long+short membership and return

# 4) Apply month by month  # iterate across months to build time series
rows = []  # list of month-level summary dicts
members = []  # list of month-level membership DataFrames
for m, df_m in panel6.groupby("month", sort=True):  # loop over each month’s cross-section
    long_ret, short_ret, mem = pick_top_bottom_with_fallback(df_m, N_LONG, N_SHORT)  # compute long/short means and membership for this month

    rows.append({  # store month-level portfolio statistics
        "month": m,  # month label (Period[M])
        "n_long": int(mem[mem["side"]=="LONG"].shape[0]),  # realized count in long bucket
        "n_short": int(mem[mem["side"]=="SHORT"].shape[0]),  # realized count in short bucket
        "long_ret_next": long_ret,  # long bucket next-month mean return
        "short_ret_next": short_ret,  # short bucket next-month mean return
        "LS_spread": (long_ret - short_ret) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # long-short spread return (1x/1x)
        "LS_5050": (0.5*(long_ret - short_ret)) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # 50/50 self-financing long-short return
    })  # end rows.append

    mem = mem[["month","permno","MOM_rt","R_next","R_next_cap","side"]].copy()  # keep only reporting columns for membership table
    members.append(mem)  # collect membership rows

step6_summary = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)  # month-level summary table sorted by time
membership = pd.concat(members, ignore_index=True) if len(members) else pd.DataFrame()  # full membership table across months (or empty)

# 5) Output series for Step 7 (match your existing variable names)  # create monthly return series for evaluation/plots
LS_spread = step6_summary.set_index("month")["LS_spread"]  # Series indexed by month: long_ret_next - short_ret_next
LS_5050   = step6_summary.set_index("month")["LS_5050"]  # Series indexed by month: 0.5*(long_ret_next - short_ret_next)

print("=== Step 6 (Top/Bottom 3) Summary ===")  # header print for console logs
print("Months with LS_5050:", int(LS_5050.notna().sum()))  # number of months with defined LS_5050 (non-missing)
print("Avg n_long / n_short:",  # label for average bucket sizes
      float(step6_summary["n_long"].mean()),  # average realized count in long bucket
      float(step6_summary["n_short"].mean()))  # average realized count in short bucket

print("\nLS_5050 mean/std:",  # label for LS_5050 summary stats
      float(LS_5050.dropna().mean()),  # mean of LS_5050 over available months
      float(LS_5050.dropna().std(ddof=1)))  # sample std (ddof=1) of LS_5050 over available months

# 6) Plots (same as before)  # visualize performance distribution and cumulative path
cum_5050_add = LS_5050.dropna().cumsum()  # additive cumulative sum of monthly LS_5050 returns

plt.figure()  # create new figure for cumulative plot
plt.plot(cum_5050_add.index.to_timestamp(), cum_5050_add.values)  # plot cumulative returns over time
plt.title("Cumulative 50/50 Long-Short (additive) — Top3/Bottom3")  # plot title
plt.xlabel("Month")  # x-axis label
plt.ylabel("Cumulative sum")  # y-axis label
plt.show()  # display cumulative plot

plt.figure()  # create new figure for histogram plot
plt.hist(pd.to_numeric(LS_5050, errors="coerce").dropna().values, bins=30)  # histogram of monthly LS_5050 returns
plt.title("Histogram — LS_5050 (Top3/Bottom3)")  # histogram title
plt.xlabel("Monthly return")  # x-axis label
plt.ylabel("Count")  # y-axis label
plt.show()  # display histogram

# Deliverables you may want to inspect  # quick previews for debugging/inspection
display(step6_summary.head(10))  # show first 10 months of summary table
display(membership.head(20))  # show first 20 membership rows

# Convenience subsets like before  # split membership by side for winner/loser buckets
QH = membership[membership["side"] == "LONG"].copy()  # winners / high momentum subset
QL = membership[membership["side"] == "SHORT"].copy()  # losers / low momentum subset


# =========================                                                          # section header
# Step 7: Evaluate Strategy Performance (continuation of your Step 6)                 # what this step does
# Uses your Step 6 outputs:                                                          # inputs section
#   - LS_spread (winner - loser spread on next-month returns)                         # input 1
#   - LS_5050   (50/50 self-financing long-short return on next-month returns)        # input 2
# (ADDED) Step-9-style describe() diagnostics (same columns pattern you used before): # what we added
#   A) LS series distribution: LS_spread, LS_5050                                     # diagnostics A
#   B) Month-level inputs (if step6_summary exists): long_ret_next, short_ret_next, LS_spread, LS_5050  # diagnostics B
#   C) Membership-level realized returns (if membership exists): R_next, R_next_cap    # diagnostics C
# =========================                                                          # section footer

def eval_series(R: pd.Series, name: str) -> dict:                                     # compute performance stats for a monthly return series
    R = R.dropna().copy()                                                             # drop missing values and work on a copy
    R = pd.to_numeric(R, errors="coerce").dropna()                                    # force numeric; drop non-parsable rows

    T = int(R.shape[0])                                                               # number of months in the series
    mean_m = float(R.mean())                                                          # average monthly return
    vol_m  = float(R.std(ddof=1))                                                     # monthly volatility (sample std)
    sharpe_m = mean_m / vol_m if vol_m > 0 else np.nan                                # monthly Sharpe (rf assumed 0)
    tstat = mean_m / (vol_m / np.sqrt(T)) if (vol_m > 0 and T > 1) else np.nan        # t-stat of mean return

    ann_ret = 12.0 * mean_m                                                           # annualized return (linear scaling)
    ann_vol = np.sqrt(12.0) * vol_m                                                   # annualized volatility
    ann_sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan                         # annualized Sharpe ratio

    can_compound = bool(((1.0 + R) > 0).all())                                        # compounding valid only if every (1+R) > 0

    out = {                                                                           # pack results into dict
        "name": name,                                                                 # series label
        "T": T,                                                                       # sample size
        "mean_monthly": mean_m,                                                       # mean monthly return
        "vol_monthly": vol_m,                                                         # std monthly return
        "sharpe_monthly": sharpe_m,                                                   # monthly Sharpe
        "tstat": tstat,                                                               # t-stat of mean
        "ann_ret": ann_ret,                                                           # annualized return
        "ann_vol": ann_vol,                                                           # annualized vol
        "ann_sharpe": ann_sharpe,                                                     # annualized Sharpe
        "min": float(R.min()) if T > 0 else np.nan,                                   # minimum monthly return
        "max": float(R.max()) if T > 0 else np.nan,                                   # maximum monthly return
        "can_compound": can_compound,                                                 # compounding feasibility flag
    }                                                                                 # end dict
    return out                                                                        # return dict

def plot_cumulative(R: pd.Series, title: str):                                        # plot additive and (if valid) compounded cumulative curves
    R = R.dropna().copy()                                                             # drop NaNs to get a clean series
    if isinstance(R.index, pd.PeriodIndex):                                            # if index is PeriodIndex (month periods)
        x = R.index.to_timestamp()                                                    # convert to timestamps for plotting
    else:                                                                             # otherwise assume date-like index
        x = pd.to_datetime(R.index)                                                   # coerce to datetime for plotting

    can_compound = bool(((1.0 + R) > 0).all())                                        # compounding validity test

    plt.figure()                                                                      # new figure for additive curve
    plt.plot(x, R.cumsum().values)                                                    # additive cumulative sum (PnL-like)
    plt.title(title + " — Cumulative (Additive Sum)")                                 # title for additive plot
    plt.xlabel("Month")                                                               # x-axis label
    plt.ylabel("Cumulative sum")                                                      # y-axis label
    plt.show()                                                                        # render additive plot

    if can_compound:                                                                  # if compounding is mathematically valid
        cumprod = (1.0 + R).cumprod() - 1.0                                           # compounded cumulative return (wealth - 1)
        plt.figure()                                                                  # new figure for compounded curve
        plt.plot(x, cumprod.values)                                                   # plot compounded curve
        plt.title(title + " — Cumulative (Compounded)")                               # title for compounded plot
        plt.xlabel("Month")                                                           # x-axis label
        plt.ylabel("Cumulative return")                                               # y-axis label
        plt.show()                                                                    # render compounded plot
    else:                                                                             # if compounding breaks due to (1+R)<=0 somewhere
        print(f"WARNING: Cannot compound {title} because some months have (1+R) <= 0.")  # warning message

def plot_hist(R: pd.Series, title: str, bins: int = 30):                              # plot histogram of monthly returns
    R = pd.to_numeric(R, errors="coerce").dropna()                                    # ensure numeric and drop NaNs
    plt.figure()                                                                      # new figure for histogram
    plt.hist(R.values, bins=bins)                                                     # histogram of returns
    plt.title("Histogram — " + title)                                                 # histogram title
    plt.xlabel("Monthly return")                                                      # x-axis label
    plt.ylabel("Count")                                                               # y-axis label
    plt.show()                                                                        # render histogram

# ------------------------------------------------------------                         # separator
# 1) Choose which LS series to evaluate                                               # section description
#    - LS_5050 is the "portfolio return" (self-financing 50/50)                        # note on LS_5050
#    - LS_spread is a spread (not a normalized portfolio return)                       # note on LS_spread
# ------------------------------------------------------------                         # separator
if "LS_5050" not in globals() or "LS_spread" not in globals():                        # ensure Step 6 outputs exist
    raise NameError("Step 7 expects LS_5050 and LS_spread from your Step 6 cell.")    # fail fast with clear error

# ------------------------------------------------------------                         # separator
# 2) Evaluate both (recommended)                                                       # section description
# ------------------------------------------------------------                         # separator
stats_5050   = eval_series(LS_5050,   "LS_5050 (50/50 self-financing)")                # compute stats for LS_5050
stats_spread = eval_series(LS_spread, "LS_spread (winners - losers spread)")          # compute stats for LS_spread

summary = pd.DataFrame([stats_5050, stats_spread])                                    # put both stats dicts into a DataFrame

print("\n=== Step 7 Summary Table ===")                                                # print header
display(summary)                                                                       # display summary table nicely

# ------------------------------------------------------------                         # separator
# (ADDED) Step 7 diagnostics table(s), like Step 9 "gross vs net"                      # diagnostics section
# ------------------------------------------------------------                         # separator

# A) Distribution of the strategy return series themselves                             # diagnostics A header
step7_check = pd.DataFrame({                                                           # build a small DataFrame of LS series
    "LS_spread": pd.to_numeric(LS_spread, errors="coerce"),                             # coerce LS_spread to numeric
    "LS_5050":   pd.to_numeric(LS_5050, errors="coerce"),                               # coerce LS_5050 to numeric
})                                                                                      # end DataFrame
print("\nStep 7 check (LS series distribution):")                                       # print header
display(step7_check.describe())                                                         # describe() of LS series

# B) If you still have step6_summary, show month-level long/short + LS                  # diagnostics B header
if "step6_summary" in globals():                                                        # only run if step6_summary exists
    cols = [c for c in ["long_ret_next", "short_ret_next", "LS_spread", "LS_5050"] if c in step6_summary.columns]  # choose available cols
    if cols:                                                                            # proceed only if at least one column exists
        print("\nStep 7 check (Top/Bottom month-level inputs):")                        # print header
        display(step6_summary[cols].apply(pd.to_numeric, errors="coerce").describe())   # describe() for month-level inputs

# C) If you still have membership, show realized returns BEFORE vs AFTER capping        # diagnostics C header
if "membership" in globals():                                                           # only run if membership exists
    cols = [c for c in ["R_next", "R_next_cap"] if c in membership.columns]             # select available columns
    if cols:                                                                            # proceed only if at least one exists
        print("\nStep 7 check (selected names: R_next vs R_next_cap):")                 # print header
        display(membership[cols].apply(pd.to_numeric, errors="coerce").describe())      # describe() for membership-level returns

# ------------------------------------------------------------                         # separator
# 3) Plot cumulative performance + histograms                                           # plotting section
# ------------------------------------------------------------                         # separator
plot_cumulative(LS_5050, "LS_5050 (50/50 self-financing)")                              # plot additive + compounded if valid for LS_5050
plot_hist(LS_5050, "LS_5050 (50/50 self-financing)", bins=30)                           # histogram for LS_5050

plot_cumulative(LS_spread, "LS_spread (spread, not normalized)")                        # plot additive + compounded if valid for LS_spread
plot_hist(LS_spread, "LS_spread (spread, not normalized)", bins=30)                     # histogram for LS_spread

# ------------------------------------------------------------                         # separator
# 4) Optional: show best/worst months for LS_5050                                       # optional diagnostics
# ------------------------------------------------------------                         # separator
tmp = LS_5050.dropna().copy()                                                           # non-missing LS_5050 months only
print("\nLS_5050 min/max:", float(tmp.min()), float(tmp.max()))                         # print min and max monthly returns
print("\nTop 10 LS_5050 months:")                                                       # header for top months
display(tmp.sort_values(ascending=False).head(10))                                      # show top 10 months
print("\nBottom 10 LS_5050 months:")                                                    # header for bottom months
display(tmp.sort_values(ascending=True).head(10))                                       # show bottom 10 months


# =========================  # section header: Step 8 strategy description
# Step 8 (Strategy 3): Inverse-volatility weighting (Enhanced)  # high-level name
# Uses ENHANCED returns (R_t1_enh) to compute sigma_rt (no look-ahead)  # sigma source
# Uses membership (from Step 6) which already has R_next and R_next_cap and side  # membership source
# Outputs: LS_5050_iv, LS_spread_iv + summary + plots/hists + Step-8-style describe tables  # outputs summary
# =========================  # end header

# ------------------------------------------------------------  # divider
# Guardrails (required inputs + required columns)  # what this block checks
# ------------------------------------------------------------  # divider
if "membership" not in globals() or "step5" not in globals():  # ensure Step 6 membership and Step 5 table exist
    raise NameError("Step 8 expects `membership` (from Step 6) and `step5` (from Step 5).")  # stop early if missing

need_cols = ["month", "permno", "side", "R_next", "R_next_cap"]  # required membership columns for this Step 8
missing_cols = [c for c in need_cols if c not in membership.columns]  # compute which required columns are missing
if len(missing_cols) > 0:  # if any required columns missing
    raise NameError(f"Step 8 expects membership to contain columns: {need_cols}. Missing: {missing_cols}")  # stop and report

if "R_t1_enh" not in step5.columns:  # ensure enhanced return column exists in step5 for sigma computation
    raise NameError("Step 8 expects step5 to contain column: 'R_t1_enh' (enhanced position returns).")  # stop and report

# ------------------------------------------------------------  # divider
# 1) Build sigma_rt from enhanced returns with no-lookahead timing  # sigma computation step
#    sigma_rt(t) = std(R_{t-12} ... R_{t-2}) via shift(2).rolling(11).std  # timing rule
# ------------------------------------------------------------  # divider
vol_base = step5[["permno", "month", "R_t1_enh"]].copy()  # keep only fields needed for sigma computation
vol_base["permno"] = pd.to_numeric(vol_base["permno"], errors="coerce").astype("Int64")  # normalize permno dtype
vol_base["R_t1_enh"] = pd.to_numeric(vol_base["R_t1_enh"], errors="coerce")  # ensure returns numeric
vol_base["month"] = pd.PeriodIndex(vol_base["month"].astype(str), freq="M")  # normalize month to PeriodIndex(M)
vol_base = vol_base.sort_values(["permno", "month"]).reset_index(drop=True)  # sort for correct rolling order

vol_base["sigma_rt"] = (  # create sigma_rt column (rolling volatility)
    vol_base.groupby("permno", group_keys=False)["R_t1_enh"]  # within each permno, take chronological return series
            .apply(lambda s: s.shift(2).rolling(window=11, min_periods=8).std(ddof=1))  # std of (t-12..t-2), require >=8 obs
)  # end sigma assignment

sigma_table = vol_base[["permno", "month", "sigma_rt"]].copy()  # keep only sigma table columns for merging

# ------------------------------------------------------------  # divider
# 2) Merge sigma onto membership (so each selected name has sigma at signal month t)  # merge step
# ------------------------------------------------------------  # divider
mem8 = membership.copy()  # working copy so original membership stays unchanged
mem8["permno"] = pd.to_numeric(mem8["permno"], errors="coerce").astype("Int64")  # normalize permno dtype
if not isinstance(mem8["month"].dtype, pd.PeriodDtype):  # if month is not already Period dtype
    mem8["month"] = pd.PeriodIndex(mem8["month"].astype(str), freq="M")  # coerce month to PeriodIndex(M)

mem8["R_next"] = pd.to_numeric(mem8["R_next"], errors="coerce")  # ensure realized next-month return numeric
mem8["R_next_cap"] = pd.to_numeric(mem8["R_next_cap"], errors="coerce")  # ensure capped next-month return numeric

mem8 = mem8.merge(sigma_table, on=["permno", "month"], how="left")  # attach sigma_rt by (permno, month)

# ------------------------------------------------------------  # divider
# 3) Inverse-vol weights, normalized within each (month, side)  # weighting step
# ------------------------------------------------------------  # divider
mem8["inv_vol"] = np.where(  # compute inverse-vol signal where sigma valid
    mem8["sigma_rt"].notna() & (mem8["sigma_rt"] > 0),  # sigma must exist and be strictly positive
    1.0 / mem8["sigma_rt"],  # inv-vol = 1/sigma
    np.nan  # otherwise invalid
)  # end inv_vol definition

mem8 = mem8[mem8["inv_vol"].notna() & mem8["R_next_cap"].notna()].copy()  # require inv_vol and realized return for weighting

mem8["w_raw"] = mem8["inv_vol"]  # raw weights = inverse volatility (pre-normalization)
w_sum = mem8.groupby(["month", "side"])["w_raw"].transform("sum")  # sum of raw weights within each month-side bucket
mem8["w_norm"] = np.where(w_sum > 0, mem8["w_raw"] / w_sum, np.nan)  # normalized weights sum to 1 within month-side

# ------------------------------------------------------------  # divider
# (ADDED) Step 8 describe table #1: position-level inputs used (MATCH screenshot columns)  # diagnostics table 1
# Columns must be: sigma_rt, w_raw, w_norm, R_next, R_next_cap  # required columns
# ------------------------------------------------------------  # divider
print("\nStep 8 check (position-level inputs used):")  # header line like your screenshot
cols_pos = ["sigma_rt", "w_raw", "w_norm", "R_next", "R_next_cap"]  # exact column order requested
display(mem8[cols_pos].apply(pd.to_numeric, errors="coerce").describe())  # describe() across all position rows used

# ------------------------------------------------------------  # divider
# 4) Weighted long/short returns per month, then compute LS series  # aggregation step
# ------------------------------------------------------------  # divider
by_ms = (  # start month-side aggregation table
    mem8.groupby(["month", "side"])  # group by signal month and side (LONG/SHORT)
        .apply(lambda g: float((g["w_norm"] * g["R_next_cap"]).sum()) if g["w_norm"].notna().any() else np.nan)  # weighted sum
        .rename("ret_next_iv")  # name the aggregated output
        .reset_index()  # convert group keys back to columns
)  # end by_ms

wide8 = by_ms.pivot(index="month", columns="side", values="ret_next_iv").sort_index()  # wide: month rows, LONG/SHORT columns
long_iv = wide8.get("LONG", pd.Series(index=wide8.index, dtype="float64"))  # long leg per month (or NaN series if missing)
short_iv = wide8.get("SHORT", pd.Series(index=wide8.index, dtype="float64"))  # short leg per month (or NaN series if missing)

LS_spread_iv = (long_iv - short_iv).rename("LS_spread_iv")  # spread = LONG - SHORT
LS_5050_iv = (0.5 * (long_iv - short_iv)).rename("LS_5050_iv")  # 50/50 self-financing = 0.5*(LONG - SHORT)

# ------------------------------------------------------------  # divider
# (ADDED) Step 8 describe table #2: month-level IV long/short + LS (MATCH screenshot columns)  # diagnostics table 2
# Columns must be: loser_bucket_ret_iv, winner_bucket_ret_iv, LS_spread_iv, LS_5050_iv  # required columns
# Note: for Strategy 3, "winner" = LONG leg, "loser" = SHORT leg  # interpretation
# ------------------------------------------------------------  # divider
step8_check_month = pd.DataFrame({  # build month-level diagnostic table
    "loser_bucket_ret_iv": pd.to_numeric(short_iv, errors="coerce"),  # loser leg return (SHORT)
    "winner_bucket_ret_iv": pd.to_numeric(long_iv, errors="coerce"),  # winner leg return (LONG)
    "LS_spread_iv": pd.to_numeric(LS_spread_iv, errors="coerce"),  # spread series
    "LS_5050_iv": pd.to_numeric(LS_5050_iv, errors="coerce"),  # 50/50 series
})  # end month-level table
print("\nStep 8 check (month-level IV winner/loser + LS):")  # header line like your screenshot
display(step8_check_month.describe())  # describe() at the month level (count/mean/std/min/quantiles/max)

# ------------------------------------------------------------  # divider
# 5) Performance summary + plots/hists (like Step 7)  # evaluation step
# ------------------------------------------------------------  # divider
summary8 = pd.DataFrame([  # one row per series evaluated
    eval_series(LS_5050_iv, "LS_5050_iv (inv-vol, 50/50, enh)"),  # evaluate 50/50 IV series
    eval_series(LS_spread_iv, "LS_spread_iv (inv-vol, spread, enh)")  # evaluate spread IV series
])  # end summary DataFrame

print("\n=== Step 8 Performance Summary (Inverse-Vol, Enhanced) ===")  # header for summary table
display(summary8)  # show summary metrics table

plot_cumulative(LS_5050_iv, "LS_5050_iv (inv-vol, 50/50, enh)")  # plot additive + compounded (or warning) for LS_5050_iv
plot_hist(LS_5050_iv, "LS_5050_iv (inv-vol, 50/50, enh)", bins=30)  # histogram for LS_5050_iv monthly returns

plot_cumulative(LS_spread_iv, "LS_spread_iv (inv-vol, spread, enh)")  # plot additive + compounded (or warning) for LS_spread_iv
plot_hist(LS_spread_iv, "LS_spread_iv (inv-vol, spread, enh)", bins=30)  # histogram for LS_spread_iv monthly returns


# =========================  # section header comment
# Step 9 (Strategy 3): Net-of-costs returns — SAME COST LOGIC as Strategy 1 Step 9  # what this block does
# But applied to the ENHANCED exit:  # clarification
#   - if HOLD: exit is exit_intrinsic  # rule 1
#   - if STOP/PROFIT: exit is exit_value (mid-based combined value)  # rule 2
# Then recompute MOM on R_t1_enh_net and re-run Strategy 2-style Top3/Bottom3 on NET.  # pipeline continuation
# PLUS: print the 3 describe() tables matching your screenshots.  # your request
# =========================  # end header comment

if "all_enh" not in globals():  # guard: require enhanced position-level table exists
    raise NameError("Step 9 expects `all_enh` (your enhanced position-level table).")  # hard fail early with clear message

# -----------------------------  # section divider
# 0) Parameters (same structure as Strategy 1 Step 9)  # parameter block
# -----------------------------  # section divider
slippage_rate_entry = 0.001  # 0.10% entry slippage (pay above mid)
exit_cost_rate      = 0.001  # 0.10% exit haircut (receive below gross exit value)
cost_bps_per_leg    = 5      # 5 bps per leg
n_legs              = 2      # straddle has 2 legs: call + put
cost_rate_total     = (cost_bps_per_leg / 10000.0) * n_legs  # convert bps to decimal and multiply by number of legs

CAP_LO, CAP_HI = -0.95, 5.0  # cap next-month returns like Step 6 (stabilize tails)

# -----------------------------  # section divider
# 1) Build net enhanced return at POSITION level (same logic as Strategy 1)  # position-level net return construction
# -----------------------------  # section divider
all_enh_9 = all_enh.copy()  # working copy so we do not overwrite your original enhanced table

for c in ["entry_mid", "exit_intrinsic", "exit_value", "R_t1_enh"]:  # list of columns we need numeric for computations/diagnostics
    if c in all_enh_9.columns:  # guard: only coerce if column exists
        all_enh_9[c] = pd.to_numeric(all_enh_9[c], errors="coerce")  # coerce to numeric; non-numeric becomes NaN

all_enh_9["exit_gross_enh"] = np.where(  # build the realized gross exit value used by the enhanced strategy
    all_enh_9["exit_reason"].astype(str).eq("HOLD"),  # condition: if we held to expiry
    all_enh_9["exit_intrinsic"],  # then gross exit is intrinsic payoff at expiry
    all_enh_9["exit_value"],  # else (STOP/PROFIT) gross exit is the mid-based combined value on exit_date
)  # end np.where

valid = (  # validity mask for net return economics
    all_enh_9["entry_mid"].notna() & (all_enh_9["entry_mid"] > 0) &  # entry premium must be positive and non-missing
    all_enh_9["exit_gross_enh"].notna() & (all_enh_9["exit_gross_enh"] >= 0)  # exit value must be non-missing and non-negative
)  # end valid mask

all_enh_9["entry_eff"] = np.where(  # effective entry execution (worse than mid)
    valid,  # compute only where row is valid
    all_enh_9["entry_mid"] * (1.0 + slippage_rate_entry),  # pay mid plus slippage
    np.nan,  # invalid -> NaN
)  # end entry_eff

all_enh_9["exit_eff"] = np.where(  # effective exit execution (worse than gross exit)
    valid,  # compute only where row is valid
    all_enh_9["exit_gross_enh"] * (1.0 - exit_cost_rate),  # haircut exit value
    np.nan,  # invalid -> NaN
)  # end exit_eff

all_enh_9["roundtrip_cost"] = np.where(  # proportional transaction cost modeled off entry premium notional
    valid,  # compute only where row is valid
    all_enh_9["entry_mid"] * cost_rate_total,  # cost rate times entry premium
    np.nan,  # invalid -> NaN
)  # end roundtrip_cost

all_enh_9["R_t1_enh_net"] = np.where(  # net enhanced return (denominator uses entry_mid for comparability)
    valid,  # compute only where row is valid
    (all_enh_9["exit_eff"] - all_enh_9["entry_eff"] - all_enh_9["roundtrip_cost"]) / all_enh_9["entry_mid"],  # net PnL / entry_mid
    np.nan,  # invalid -> NaN
)  # end R_t1_enh_net

print("Step 9 check (enh gross vs enh net):")  # quick diagnostic header
display(all_enh_9[["entry_mid", "exit_gross_enh", "R_t1_enh", "R_t1_enh_net"]].describe())  # gross-vs-net subset describe

print("\nStep 9 check (position-level economics, expanded):")  # screenshot #1 header
pos_econ = all_enh_9[["entry_mid", "exit_gross_enh", "R_t1_enh", "entry_eff", "exit_eff", "roundtrip_cost", "R_t1_enh_net"]].copy()  # select columns
pos_econ = pos_econ.rename(columns={  # rename columns to match your screenshot labels exactly
    "exit_gross_enh": "exit_intrinsic",  # screenshot column name (here it means realized enhanced exit gross)
    "R_t1_enh": "R_t1",  # screenshot column name for gross return
    "R_t1_enh_net": "R_t1_net",  # screenshot column name for net return
})  # end rename
display(pos_econ.describe())  # describe table with exact screenshot column names

# -----------------------------  # section divider
# 2) Step 5 on NET: MOM_rt_net from R_t1_enh_net (no look-ahead)  # momentum recomputation using net returns
# -----------------------------  # section divider
step5_net = all_enh_9.copy()  # start from position-level net table
step5_net["permno"] = pd.to_numeric(step5_net["permno"], errors="coerce").astype("Int64")  # standardize permno type
step5_net["R_t1_enh_net"] = pd.to_numeric(step5_net["R_t1_enh_net"], errors="coerce")  # ensure net return is numeric

step5_net["_month_ts"] = pd.PeriodIndex(step5_net["month"].astype(str), freq="M").to_timestamp()  # build sortable timestamp month key
step5_net = step5_net.sort_values(["permno", "_month_ts"]).reset_index(drop=True)  # sort by permno then time for rolling

step5_net["MOM_rt_net"] = (  # compute momentum signal on net returns
    step5_net.groupby("permno", group_keys=False)["R_t1_enh_net"]  # within each permno, take net returns series
            .apply(lambda s: s.shift(2).rolling(window=11, min_periods=8).mean())  # mean of (t-12..t-2), no look-ahead
)  # end MOM_rt_net

step5_net = step5_net.drop(columns=["_month_ts"])  # cleanup helper column

# -----------------------------  # section divider
# 3) Step 6 on NET: Top3/Bottom3 with fallback (reusing your pick_top_bottom_with_fallback)  # portfolio construction on net
# -----------------------------  # section divider
if "pick_top_bottom_with_fallback" not in globals():  # guard: must have the selector defined in your Step 6 cell
    raise NameError("Step 9 expects `pick_top_bottom_with_fallback` from Step 6 (Top/Bottom selection).")  # fail if missing

panel6_net = step5_net[["month", "permno", "R_t1_enh_net", "MOM_rt_net"]].copy()  # panel needed for ranking + next-month eval
panel6_net["permno"] = pd.to_numeric(panel6_net["permno"], errors="coerce").astype("Int64")  # permno clean
panel6_net["R_t1_enh_net"] = pd.to_numeric(panel6_net["R_t1_enh_net"], errors="coerce")  # net return clean
panel6_net["MOM_rt_net"] = pd.to_numeric(panel6_net["MOM_rt_net"], errors="coerce")  # net momentum clean
panel6_net["month"] = pd.PeriodIndex(panel6_net["month"].astype(str), freq="M")  # month as PeriodIndex for grouping/indexing

panel6_net = panel6_net.sort_values(["permno", "month"]).reset_index(drop=True)  # sort for shift(-1)
panel6_net["R_next_net"] = panel6_net.groupby("permno")["R_t1_enh_net"].shift(-1)  # next-month realized net return
panel6_net["R_next_net_cap"] = panel6_net["R_next_net"].clip(lower=CAP_LO, upper=CAP_HI)  # capped next-month return

panel6_net = panel6_net.dropna(subset=["permno", "month", "MOM_rt_net"]).copy()  # keep only rows with usable MOM for ranking

print("\nStep 9 check (membership-level, net):")  # screenshot #3 header
mem9_net = panel6_net[["R_next_net", "R_next_net_cap"]].copy()  # membership-level (universe-level) next-month returns
mem9_net["R_next_net"] = pd.to_numeric(mem9_net["R_next_net"], errors="coerce")  # ensure numeric
mem9_net["R_next_net_cap"] = pd.to_numeric(mem9_net["R_next_net_cap"], errors="coerce")  # ensure numeric
display(mem9_net.describe())  # describe() with exact screenshot column names

rows9 = []  # will store month-level long/short legs and LS series
members9 = []  # will store selected membership rows (Top3/Bottom3) for inspection

N_LONG  = globals().get("N_LONG", 3)  # long side size (fallback to 3)
N_SHORT = globals().get("N_SHORT", 3)  # short side size (fallback to 3)

for m, df_m in panel6_net.groupby("month", sort=True):  # iterate month-by-month over the ranking universe
    df_tmp = df_m.rename(columns={  # rename columns to match what pick_top_bottom_with_fallback expects
        "MOM_rt_net": "MOM_rt",  # expected signal name
        "R_next_net": "R_next",  # expected realized next-month return name
        "R_next_net_cap": "R_next_cap",  # expected capped next-month return name
    }).copy()  # make a copy to avoid modifying panel6_net

    long_ret, short_ret, mem = pick_top_bottom_with_fallback(df_tmp, N_LONG, N_SHORT)  # compute long/short legs + membership rows

    rows9.append({  # store month-level outputs (this is what you want in screenshot #2)
        "month": m,  # signal month
        "n_long": int(mem[mem["side"] == "LONG"].shape[0]),  # realized number of longs (after fallback)
        "n_short": int(mem[mem["side"] == "SHORT"].shape[0]),  # realized number of shorts (after fallback)
        "long_ret_next_net": long_ret,  # average (capped) next-month net return of long basket
        "short_ret_next_net": short_ret,  # average (capped) next-month net return of short basket
        "LS_spread_net": (long_ret - short_ret) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # spread
        "LS_5050_net": (0.5 * (long_ret - short_ret)) if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,  # 50/50
    })  # end dict

    mem_keep = mem[["month", "permno", "MOM_rt", "R_next", "R_next_cap", "side"]].copy()  # keep clean membership columns
    mem_keep = mem_keep.rename(columns={  # rename to net-specific names for clarity
        "MOM_rt": "MOM_rt_net",  # net momentum
        "R_next": "R_next_net",  # net next-month return
        "R_next_cap": "R_next_net_cap",  # capped net next-month return
    })  # end rename
    members9.append(mem_keep)  # collect membership

step9_summary = pd.DataFrame(rows9).sort_values("month").reset_index(drop=True)  # month-level summary table
membership_net = pd.concat(members9, ignore_index=True) if len(members9) else pd.DataFrame()  # membership table (Top/Bottom only)

LS_spread_net = step9_summary.set_index("month")["LS_spread_net"]  # LS spread series indexed by month
LS_5050_net   = step9_summary.set_index("month")["LS_5050_net"]  # LS 50/50 series indexed by month

print("\nStep 9 check (month-level inputs, net):")  # screenshot #2 header
step9_month_inputs_net = step9_summary.set_index("month")[["long_ret_next_net", "short_ret_next_net", "LS_spread_net", "LS_5050_net"]].copy()  # exact columns
step9_month_inputs_net["long_ret_next_net"] = pd.to_numeric(step9_month_inputs_net["long_ret_next_net"], errors="coerce")  # numeric safety
step9_month_inputs_net["short_ret_next_net"] = pd.to_numeric(step9_month_inputs_net["short_ret_next_net"], errors="coerce")  # numeric safety
step9_month_inputs_net["LS_spread_net"] = pd.to_numeric(step9_month_inputs_net["LS_spread_net"], errors="coerce")  # numeric safety
step9_month_inputs_net["LS_5050_net"] = pd.to_numeric(step9_month_inputs_net["LS_5050_net"], errors="coerce")  # numeric safety
display(step9_month_inputs_net.describe())  # describe() with exact screenshot column names

print("\n=== Step 9 (Net, Enhanced) Top/Bottom 3 Summary ===")  # optional summary header
display(step9_summary.head(10))  # show first 10 months summary
display(membership_net.head(20))  # show first 20 membership rows

# -----------------------------  # section divider
# 4) Step 7-style evaluation + plots/hists (both series)  # performance evaluation and charts
# -----------------------------  # section divider
if "eval_series" not in globals():  # guard: need your Step 7 eval function
    raise NameError("Step 9 expects `eval_series` from Step 7.")  # fail if missing
if "plot_cumulative" not in globals():  # guard: need your Step 7 plotting function (handles additive + compounded/warning)
    raise NameError("Step 9 expects `plot_cumulative` from Step 7.")  # fail if missing
if "plot_hist" not in globals():  # guard: need your Step 7 histogram function
    raise NameError("Step 9 expects `plot_hist` from Step 7.")  # fail if missing

summary9 = pd.DataFrame([  # build performance summary table
    eval_series(LS_5050_net,   "LS_5050_net (Top3/Bottom3, net, enh)"),  # evaluate 50/50 net series
    eval_series(LS_spread_net, "LS_spread_net (spread, net, enh)"),  # evaluate spread net series
])  # end summary DataFrame

print("\n=== Step 9 Net Performance Summary (Enhanced) ===")  # header for summary table
display(summary9)  # display summary metrics

plot_cumulative(LS_5050_net, "LS_5050_net (Top3/Bottom3, net, enh)")  # cumulative plots (additive + compounded-or-warning)
plot_hist(LS_5050_net, "LS_5050_net (Top3/Bottom3, net, enh)", bins=30)  # histogram of monthly net 50/50 returns

plot_cumulative(LS_spread_net, "LS_spread_net (spread, net, enh)")  # cumulative plots (additive + compounded-or-warning)
plot_hist(LS_spread_net, "LS_spread_net (spread, net, enh)", bins=30)  # histogram of monthly net spread returns

