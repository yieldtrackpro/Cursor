#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DYPS Lite — Robust Self IV Rank + Percentile (v7.3)
===================================================

- Robust ATM 30D IV from Polygon snapshot:
  * Prefer direct 30D ATM (calls & puts).
  * √T interpolation from 7/14/21 if 30D missing.
  * Median blend with sanity checks when sources disagree.
  * Clip to [0.05, 2.00]. Carry-forward up to 5 business days if needed.

- Self IV Rank / Percentile vs own 1y history (daily):
  * p_raw: Hazen empirical on IV (Polygon-sourced only by default).
  * p_log: Hazen empirical on log(IV) using strictly positive IVs.
  * p_ew : Exponentially-weighted empirical with time-aware decay (half-life in days).
  * p_final = 0.5*p_raw + 0.3*p_log + 0.2*p_ew.
  * Require N >= 30 valid history points (Polygon-sourced) for a score by default.
    Use --allow-bootstrap to relax to N >= 10 and allow carry-forward IV for rank only.

- Output fields: (only the ones you asked to keep)
  fund, Group, Underlying, Ex-Date, Pay Date,
  Price, Price ($), Price as of,
  Est. Dividend, Est. Dividend ($), Est. Range, EstLow, EstHigh, Fwd Yield (Ann.),
  IV_Rank_Self_0_100, IV_Percentile_Self_0_1,
  RV30, IV-RV Gap (%), IV_RV_Gap_Pct,
  Hist Range (13m/52w), HistLow, HistHigh, HistN, HistLabel, Days to Ex

Run:
  python dyps_ivrank_robust_v7_2.py --save ytp_output.csv --polygon-key pk_live_xxx
  # or set POLYGON_API_KEY env, or paste key in POLYGON_API_KEY_DEFAULT.

Deps:
  pip install -U pandas numpy requests requests-cache openpyxl
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
import requests_cache

# ---------- USER: paste your Polygon key here if you like ----------
POLYGON_API_KEY_DEFAULT = ""  # e.g., "pk_live_...."

# ---------- Config ----------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}

CACHE_EXPIRE_SEC = 1800
RATE_LIMIT_SEC, JITTER = 0.30, 0.20

RANGE_PCT_DEFAULT = 0.10
IVHIST_PATH_DEFAULT = "ytp_iv_history.csv"
CARRY_FWD_MAX_BUSINESS_DAYS = 5

# Annualization
FREQ_BY_GROUP = {"Weekly": 52, "A": 13, "B": 13, "C": 13, "D": 13}

# Roster
FALLBACK_GROUPS = {
    "Weekly": ["CHPY","GPTY","LFGY","QDTY","RDTY","SDTY","ULTY","YMAG","YMAX"],
    "A": ["BRKC","CRSH","FEAT","FIVY","GOOY","OARK","RBLY","SNOY","TSLY","TSMY","XOMO","YBIT"],
    "B": ["BABO","DIPS","FBY","GDXY","JPMO","MARO","MRNY","NVDY","PLTY","PYPY"],
    "C": ["ABNY","AMDY","CONY","CVNY","DRAY","FIAT","HOOY","MSFO","NFLY"],
    "D": ["AIYY","AMZY","APLY","DISO","MSTY","SMCY","WNTR","XYZY","YQQQ"],
}
GROUP_OF = {s: g for g, arr in FALLBACK_GROUPS.items() for s in arr}
WHITELIST = set(GROUP_OF.keys())

UNDERLYING: Dict[str, Optional[str]] = {
    # A
    "TSLY":"TSLA","CRSH":"TSLA","GOOY":"GOOGL","OARK":"ARKK","RBLY":"RBLX","SNOY":"SNOW",
    "TSMY":"TSM","XOMO":"XOM","BRKC":"BRK-B","YBIT":"BITO","FEAT":None,"FIVY":None,
    # B
    "NVDY":"NVDA","DIPS":"NVDA","PLTY":"PLTR","FBY":"META","MRNY":"MRNA","PYPY":"PYPL",
    "MARO":"MARA","JPMO":"JPM","GDXY":"GDX","BABO":"BABA",
    # C
    "AMDY":"AMD","CONY":"COIN","CVNY":"CVNA","HOOY":"HOOD","MSFO":"MSFT","NFLY":"NFLX",
    "ABNY":"ABNB","DRAY":"DKNG","FIAT":"COIN",
    # D
    "AMZY":"AMZN","APLY":"AAPL","DISO":"DIS","MSTY":"MSTR","SMCY":"SMCI","AIYY":"AI",
    "YQQQ":"QQQ","WNTR":"MSTR","XYZY":"QQQ",
    # Weeklies
    "CHPY":"SPY","GPTY":"QQQ","LFGY":"SPY","QDTY":"QQQ","RDTY":"IWM","SDTY":"SPY","ULTY":"SPY","YMAG":"SPY","YMAX":"QQQ",
}

ANCHOR_EX = {"A": date(2025,1,23), "B": date(2025,1,3), "C": date(2025,1,9), "D": date(2025,1,16), "Weekly": date(2025,1,3)}

# ---------- Utils ----------
def _sleep():
    time.sleep(max(0.0, RATE_LIMIT_SEC + random.uniform(-JITTER, JITTER)))

def normalize_ticker(s: str) -> str:
    if not s: return ""
    return (s.replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")
             .replace("\u00A0"," ").strip().upper())

def fmt_asof(ts: Optional[int]) -> str:
    if ts is None: return ""
    try: return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception: return ""

def is_weekly(fund: str) -> bool:
    return GROUP_OF.get(fund, "") == "Weekly"

def next_ex_date_for_group(group: str, today: date) -> date:
    if group not in ANCHOR_EX: raise ValueError(f"Unknown group: {group}")
    anchor = ANCHOR_EX[group]; step = 7 if group=="Weekly" else 28
    if today <= anchor: return anchor
    k = ((today - anchor).days + step - 1)//step
    return anchor + timedelta(days=k*step)

def next_pay_date_from_ex(ex: date) -> date:
    d = ex + timedelta(days=1)
    if d.weekday() == 5: return d + timedelta(days=2)
    if d.weekday() == 6: return d + timedelta(days=1)
    return d

# ---------- Yahoo ----------
def yahoo_quote_batch_prices(session, symbols: List[str]):
    price_map = {s: None for s in symbols}
    asof_map   = {s: None for s in symbols}
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i+40]
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        try:
            j = session.get(url, headers=HEADERS, params={"symbols": ",".join(chunk)}, timeout=20).json()
            for q in j.get("quoteResponse", {}).get("result", []):
                sym = normalize_ticker(q.get("symbol",""))
                if sym not in price_map: continue
                def num(x):
                    try: return float(x)
                    except Exception: return None
                rmp = num(q.get("regularMarketPrice")); rmt = q.get("regularMarketTime")
                if rmp is not None:
                    price_map[sym]=rmp; asof_map[sym]=rmt; continue
                pmp = num(q.get("postMarketPrice")); pmt = q.get("postMarketTime")
                if pmp is not None:
                    price_map[sym]=pmp; asof_map[sym]=pmt or rmt; continue
                prev = num(q.get("regularMarketPreviousClose"))
                if prev is not None:
                    price_map[sym]=prev; asof_map[sym]=rmt; continue
                bid = num(q.get("bid")); ask = num(q.get("ask"))
                if bid and ask and bid>0 and ask>0:
                    price_map[sym]=(bid+ask)/2; asof_map[sym]=rmt
            _sleep()
        except Exception:
            _sleep()
    return price_map, asof_map

def yahoo_chart_hist_adj(session, symbol, rng="1y", interval="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        j = session.get(url, headers=HEADERS, params={"range": rng, "interval": interval}, timeout=20).json()
        res = j.get("chart", {}).get("result", [])
        if not res: return [], []
        ts = res[0].get("timestamp", []) or []
        adj = (res[0].get("indicators", {}) or {}).get("adjclose", [{}])[0].get("adjclose", [])
        return ts, adj if adj else []
    except Exception:
        return [], []

def yahoo_chart_with_divs(session, symbol, rng="18mo", interval="1d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        j = session.get(url, headers=HEADERS,
                        params={"range": rng, "interval": interval, "events":"div"},
                        timeout=20).json()
        res = j.get("chart", {}).get("result", [])
        if not res: return [], [], []
        ts = res[0].get("timestamp", []) or []
        q  = res[0].get("indicators", {}).get("quote", [{}])[0]
        closes = q.get("close", []) if q else []
        divs = (res[0].get("events", {}) or {}).get("dividends", {}) or {}
        out = []
        for _, v in divs.items():
            amt = v.get("amount"); t = v.get("date") or v.get("ts") or v.get("timestamp")
            if amt is not None and t is not None: out.append((int(t), float(amt)))
        out.sort(key=lambda x: x[0])
        return ts, closes, out
    except Exception:
        return [], [], []

def yahoo_chart_last_close(session, symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        j = session.get(url, headers=HEADERS, params={"range":"5d","interval":"1d"}, timeout=20).json()
        res = j.get("chart", {}).get("result", [])
        if not res: return None, None
        closes = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        ts     = res[0].get("timestamp", [])
        pairs = [(c,t) for c,t in zip(closes, ts) if c is not None]
        if pairs:
            c, t = pairs[-1]
            return float(c), int(t) if t is not None else None
    except Exception:
        pass
    return None, None

def fill_missing_prices_with_chart(session, symbols, price_map, asof_map):
    for s in symbols:
        if price_map.get(s) is None:
            px, ts = yahoo_chart_last_close(session, s)
            if px is not None:
                price_map[s]=px; asof_map[s]=ts
            _sleep()

# ---------- Polygon ----------
POLY_SYMBOL_FIX = {"BRK-B":"BRK.B"}
def _poly_symbol(sym: str) -> str:
    return POLY_SYMBOL_FIX.get(sym, sym)

def _clean_api_key(k: str) -> str:
    if not k: return ""
    return "".join(ch for ch in k.strip() if ch.isalnum() or ch in "_-")

class PolygonClient:
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        self.api_key = api_key
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
    def _get(self, path: str, params: dict) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = f"{self.base}{path}"
        for attempt in range(3):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            time.sleep(1 + attempt)
        r.raise_for_status()
    def _paged(self, path: str, params: dict) -> List[dict]:
        params = dict(params or {}); params.setdefault("limit", 250)
        out, safety = [], 0
        while True:
            data = self._get(path, params)
            out.extend(data.get("results", []) or [])
            nxt = data.get("next_url") or ""
            cursor = data.get("next_page_token") or None
            if nxt and not cursor:
                qs = parse_qs(urlparse(nxt).query)
                cursor = (qs.get("cursor") or [None])[0]
            if not cursor: break
            params = {"cursor": cursor}
            safety += 1
            if safety > 50: break
        return out
    def snapshot_underlying(self, underlying: str, contract_type: Optional[str]) -> List[dict]:
        path = f"/v3/snapshot/options/{_poly_symbol(underlying)}"
        params = {}
        if contract_type in ("call","put"):
            params["contract_type"] = contract_type
        return self._paged(path, params)

def _dte_poly(r: dict) -> Optional[int]:
    det = r.get("details") or {}
    exp = det.get("expiration_date")
    if not exp: return None
    try:
        expd = datetime.fromisoformat(exp.replace("Z","")).date()
        return (expd - date.today()).days
    except Exception:
        return None

def _iv_poly(r: dict) -> Optional[float]:
    g = r.get("greeks") or {}
    iv = g.get("iv") if g.get("iv") not in (None, 0) else g.get("implied_volatility")
    try:
        return float(iv) if iv not in (None, 0) else None
    except Exception:
        return None

def _nearest_atm_poly(rows: List[dict]) -> Optional[dict]:
    best, best_key = None, 1e9
    for r in rows:
        greeks = r.get("greeks") or {}
        delta = greeks.get("delta")
        if delta is None:
            det = r.get("details") or {}
            strike = det.get("strike_price")
            und   = (r.get("underlying_asset") or {}).get("price")
            if not (strike and und): continue
            key = abs(float(strike)/float(und) - 1.0)
        else:
            key = abs(abs(delta) - 0.5)
        if key < best_key:
            best, best_key = r, key
    return best

def _bucket_atm_poly(rows: List[dict], target: int, tol=5) -> Optional[dict]:
    r2 = [r for r in rows if _dte_poly(r) is not None and abs(_dte_poly(r)-target) <= tol]
    if not r2: return None
    return _nearest_atm_poly(r2)

def _median_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    arr = [v for v in vals if v is not None and np.isfinite(v)]
    if not arr: return None
    return float(np.median(arr))

def compute_iv30_polygon(pc: PolygonClient, underlying: str) -> Tuple[Optional[float], str]:
    """Return best-effort IV30 and source tag."""
    calls = pc.snapshot_underlying(underlying, "call")
    puts  = pc.snapshot_underlying(underlying, "put")
    if not calls and not puts:
        return None, "none"
    # collect ATM at 7/14/21/30 for both sides
    def collect(rows, t):
        rc = _bucket_atm_poly(rows, t); return _iv_poly(rc) if rc else None
    atm = {}
    for t in (7,14,21,30):
        atm[f"c{t}"] = collect(calls, t)
        atm[f"p{t}"] = collect(puts,  t)
    # primary 30D: median of call/put 30D
    direct30 = _median_ignore_none([atm["c30"], atm["p30"]])
    # √T interpolation from available 7/14/21 (combine call/put medians per tenor)
    pts = []
    for t in (7,14,21):
        val = _median_ignore_none([atm[f"c{t}"], atm[f"p{t}"]])
        if val is not None and val>0:
            pts.append((t, val))
    interp30 = None
    if pts:
        X = np.array([math.sqrt(t) for t,_ in pts], dtype=float)
        y = np.array([v for _,v in pts], dtype=float)
        try:
            a, b = np.polyfit(X, y, 1)
            est = float(a*math.sqrt(30.0) + b)
            if np.isfinite(est) and est>0: interp30 = float(np.clip(est, 0.05, 2.00))
        except Exception:
            pass
    # choose with consistency checks
    cand = [v for v in [direct30, interp30] if v is not None and np.isfinite(v)]
    if not cand:
        return None, "none"
    if len(cand) == 1:
        iv = float(np.clip(cand[0], 0.05, 2.00))
        return iv, ("polygon_direct" if direct30 is not None else "polygon_interpolated")
    # both present → reconcile
    d, i = direct30, interp30
    if d is not None and i is not None:
        rel = abs(d - i) / max(1e-9, (d + i) / 2.0)
        if rel <= 0.30:
            iv = float(np.clip(0.5*(d+i), 0.05, 2.00)); return iv, "polygon_blend"
        else:
            # take median across call, put, and interp if we have both sides
            med_30 = _median_ignore_none([atm["c30"], atm["p30"], interp30])
            if med_30 is not None:
                iv = float(np.clip(med_30, 0.05, 2.00)); return iv, "polygon_median"
            iv = float(np.clip(d if abs(d) < abs(i) else i, 0.05, 2.00)); return iv, "polygon_disagree"
    # fallback
    iv = float(np.clip(cand[0], 0.05, 2.00))
    return iv, ("polygon_direct" if direct30 is not None else "polygon_interpolated")

# ---------- IV history upsert & ranks ----------
def _business_days_between(d1: date, d2: date) -> int:
    if d2 < d1: d1, d2 = d2, d1
    days = 0
    cur = d1
    while cur < d2:
        if cur.weekday() < 5:
            days += 1
        cur += timedelta(days=1)
    return days

def load_ivhist(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for c in ["date","underlying","iv30","src"]:
                if c not in df.columns: df[c] = np.nan
            return df[["date","underlying","iv30","src"]].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=["date","underlying","iv30","src"])

def upsert_ivhist(path: str, today_iso: str, rows: List[dict]):
    if not rows: return
    df = load_ivhist(path)
    new = pd.DataFrame(rows)
    # drop duplicates for today+underlying from old, then append new
    keys = set((r["date"], r["underlying"]) for _, r in new.iterrows())
    if not df.empty:
        mask = df.apply(lambda r: (r["date"], r["underlying"]) not in keys, axis=1)
        df = df[mask]
    all_ = pd.concat([df, new], ignore_index=True)
    all_.to_csv(path, index=False)
    logging.info(f"IV history upsert → {path} (+{len(rows)} rows)")

def hazen_percentile(values: np.ndarray, x: float) -> Optional[float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0: return None
    vals = np.sort(vals)
    # rank r for x (1-indexed), using right-side for ties
    r = np.searchsorted(vals, x, side="right")  # count <= x
    n = len(vals)
    p = (r - 0.5) / n
    return float(min(max(p, 0.0), 1.0))

def ew_percentile(values: np.ndarray, x: float, half_life_days: float = 30.0, dates: Optional[np.ndarray] = None) -> Optional[float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0: return None
    n = len(vals)
    # Time-aware weights: newest has highest weight
    if dates is not None and len(dates) == n:
        try:
            ts = pd.to_datetime(pd.Series(dates), errors="coerce")
            # Use the most recent timestamp as reference
            t_ref = ts.max()
            # age in days as float
            age_days = (t_ref - ts).dt.total_seconds() / 86400.0
            lam = math.log(2.0) / max(1e-9, half_life_days)
            w = np.exp(-lam * age_days.to_numpy())
        except Exception:
            # fallback to index-based recency if timestamps fail
            idx = np.arange(n)
            lam = math.log(2.0) / max(1e-9, half_life_days)
            w = np.exp(lam * (idx - (n-1)))
    else:
        idx = np.arange(n)  # 0..n-1 old->new
        lam = math.log(2.0) / max(1e-9, half_life_days)
        w = np.exp(lam * (idx - (n-1)))  # max at the end
    w = w / w.sum()
    # weighted CDF at x
    mask_le = vals <= x
    p = float((w * mask_le).sum())
    return float(min(max(p, 0.0), 1.0))

def compute_self_iv_percentiles(ivhist_df: pd.DataFrame, und: str, iv_today: float,
                                lookback_days: int = 365, min_points: int = 30,
                                polygon_only: bool = True, ew_half_life_days: float = 30.0) -> Tuple[Optional[float], Optional[float]]:
    if iv_today is None or not np.isfinite(iv_today) or iv_today <= 0:
        return None, None
    if ivhist_df.empty:
        return None, None
    u_sym = _poly_symbol(und)
    df = ivhist_df[ivhist_df["underlying"] == u_sym].copy()
    if df.empty: return None, None
    # parse and filter by lookback window
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]
    if polygon_only and "src" in df.columns:
        df = df[df["src"].astype(str).str.startswith("polygon")]
    if df.empty:
        return None, None
    # de-dup per date and sort chronologically
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    # numeric and sanity clip
    v_series = pd.to_numeric(df["iv30"], errors="coerce").astype(float)
    v_series = v_series.where(np.isfinite(v_series))
    v_series = v_series.clip(lower=0.0000001, upper=10.0)
    vals = v_series.dropna().to_numpy()
    if vals.size < min_points:
        return None, None
    # three flavors
    p_raw = hazen_percentile(vals, iv_today)
    vals_pos = vals[vals > 0]
    p_log = hazen_percentile(np.log(vals_pos), math.log(iv_today)) if (iv_today > 0 and vals_pos.size > 0) else None
    p_ew  = ew_percentile(vals, iv_today, half_life_days=ew_half_life_days, dates=df["date"].to_numpy())
    parts = [p for p in [p_raw, p_log, p_ew] if p is not None]
    if not parts: return None, None
    # Blend
    w_raw, w_log, w_ew = 0.5, 0.3, 0.2
    p_final = (w_raw*(p_raw if p_raw is not None else 0.0) +
               w_log*(p_log if p_log is not None else 0.0) +
               w_ew *(p_ew  if p_ew  is not None else 0.0))
    # Clamp
    p_final = float(min(max(p_final, 0.0), 1.0))
    return float(p_final*100.0), p_final

# ---------- Simple dividend forecast (unchanged from your lite pipeline) ----------
PARAMS_WEEKLY  = dict(K=8, w=0.80, L=20, beta=0.70, clamp=0.12)
PARAMS_MONTHLY = dict(K=4, w=0.80, L=30, beta=0.80, clamp=0.15)

class Event:
    __slots__=("ts","amt","pre_close","y")
    def __init__(self, ts:int, amt:float, pre_close:float):
        self.ts=ts; self.amt=amt; self.pre_close=pre_close; self.y=amt/pre_close

def _val_on_or_before(target_ts:int, ts_list: List[int], vals: List[Optional[float]])->Optional[float]:
    cand=[(t,v) for t,v in zip(ts_list,vals) if v is not None and t<=target_ts]
    if not cand: return None
    return float(cand[-1][1])

def build_events(ts: List[int], closes: List[Optional[float]], divs: List[Tuple[int,float]])->List[Event]:
    out=[]
    for t, amt in divs:
        pre=_val_on_or_before(t-2*60*60, ts, closes)
        if pre and pre>0 and amt and amt>0:
            out.append(Event(int(t), float(amt), float(pre)))
    out.sort(key=lambda e:e.ts)
    return out

def robust_filter_yields(ys: List[float], z_mad: float=3.0)->List[float]:
    if not ys: return ys
    med = float(np.median(ys))
    mad = float(np.median([abs(y-med) for y in ys])) or 1e-9
    keep=[]
    for y in ys:
        z = 0.6745 * (y - med) / mad
        if abs(z) <= z_mad: keep.append(y)
    return keep if keep else ys

def predict_hist_mom_simple(prev_events: List[Event], ts: List[int], closes: List[Optional[float]],
                            ex_ts:int, K:int, w:float, L:int, beta:float, clamp:float)->Optional[float]:
    if not prev_events: return None
    used = prev_events[-K:] if len(prev_events)>=K else prev_events[:]
    ys = robust_filter_yields([e.y for e in used], z_mad=3.0)
    y_last = ys[-1]
    y_hist = w*y_last + (1.0-w)*float(np.median(ys))
    now_idx = max([i for i,t in enumerate(ts) if t<=ex_ts], default=None)
    if now_idx is None: return y_hist
    now_price = closes[now_idx]
    back_ts = ex_ts - L*24*60*60
    back_idx = max([i for i,t in enumerate(ts) if t<=back_ts], default=None)
    if back_idx is None: return y_hist
    past_price = closes[back_idx]
    if past_price is None or past_price<=0 or now_price is None or now_price<=0: return y_hist
    m = max(-clamp, min(clamp, now_price/past_price - 1.0))
    return y_hist * (1.0 + beta*m)

def per_period_yield_from_iv(iv: Optional[float], group: str) -> Optional[float]:
    if iv is None or not np.isfinite(iv): return None
    return (iv/52.0) if group=="Weekly" else (iv/13.0)

def historical_dividend_range(events: List[Event], group: str, asof_ts: int) -> Tuple[Optional[float], Optional[float], int, str]:
    used = [e.amt for e in events if e.ts <= asof_ts]
    if not used: return None, None, 0, ("52w" if group=="Weekly" else "13m")
    s = pd.Series(used, dtype="float64")
    low = float(s.quantile(0.15)); high= float(s.quantile(0.85))
    return low, high, int(len(used)), ("15-85pct_52w" if group=="Weekly" else "15-85pct_13m")

# ---------- Key resolution ----------
def resolve_polygon_key(args) -> Tuple[str, str]:
    if args.polygon_key:
        return _clean_api_key(args.polygon_key), "CLI --polygon-key"
    env = os.getenv("POLYGON_API_KEY", "")
    if env:
        return _clean_api_key(env), "env POLYGON_API_KEY"
    if POLYGON_API_KEY_DEFAULT:
        return _clean_api_key(POLYGON_API_KEY_DEFAULT), "POLYGON_API_KEY_DEFAULT const"
    return "", "none"

# ---------- Runner ----------
def run(save_path: Optional[str], range_pct: float,
        polygon_key: str, polygon_base_url: str,
        ivhist_path: str, allow_no_polygon: bool, allow_bootstrap: bool) -> pd.DataFrame:

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    session = requests_cache.CachedSession("ytp_cache_robust", backend="sqlite", expire_after=CACHE_EXPIRE_SEC)
    session.headers.update(HEADERS)

    if not polygon_key and not allow_no_polygon:
        logging.error("Polygon API key required. Provide --polygon-key, env POLYGON_API_KEY, or POLYGON_API_KEY_DEFAULT; "
                      "or run with --allow-fallback-if-no-polygon (no IV ranks).")
        raise SystemExit(2)
    if polygon_key:
        logging.info("Polygon key detected; IV ranks will be computed from Polygon IV30.")

    today = date.today(); today_iso = today.isoformat()
    ex_by_group = {g: next_ex_date_for_group(g, today) for g in ["A","B","C","D","Weekly"]}
    pay_by_group= {g: next_pay_date_from_ex(ex_by_group[g]) for g in ex_by_group}
    tickers = sorted(WHITELIST)

    # Prices
    logging.info("Fetching quotes for symbols…")
    price_map, asof_map = yahoo_quote_batch_prices(session, tickers)
    fill_missing_prices_with_chart(session, tickers, price_map, asof_map)

    # Polygon IV per underlying
    unds = sorted({_poly_symbol(u) for u in UNDERLYING.values() if u})
    und_rows=[]; coverage = {"polygon_direct":0,"polygon_interpolated":0,"polygon_blend":0,"polygon_median":0,
                             "polygon_disagree":0,"carried_forward":0,"none":0}
    pc = PolygonClient(polygon_key, polygon_base_url) if polygon_key else None

    # Load history once for carry-forward logic
    ivhist = load_ivhist(ivhist_path)

    for und in unds:
        iv = None; src = "none"; carried = False
        try:
            if pc:
                iv, src = compute_iv30_polygon(pc, und)
            if iv is None and polygon_key:
                # try carry-forward (<=5 business days)
                dfu = ivhist[ivhist["underlying"] == _poly_symbol(und)]
                if not dfu.empty:
                    dfu = dfu.sort_values("date")
                    last = dfu.iloc[-1]
                    try:
                        d_last = datetime.strptime(str(last["date"]), "%Y-%m-%d").date()
                    except Exception:
                        try:
                            d_last = pd.to_datetime(last["date"]).date()
                        except Exception:
                            d_last = date.today() - timedelta(days=999)
                    age = _business_days_between(d_last, date.today())
                    if age <= CARRY_FWD_MAX_BUSINESS_DAYS:
                        iv = float(last["iv30"]) if pd.notna(last["iv30"]) else None
                        if iv is not None and 0.05 <= iv <= 2.00:
                            src = "carried_forward"; carried = True
            if iv is None:
                coverage["none"] += 1
            else:
                coverage[src] = coverage.get(src, 0) + 1
            und_rows.append({"underlying": _poly_symbol(und), "iv30": float(iv) if iv is not None else np.nan, "src": src})
        except Exception as e:
            logging.warning(f"IV30 failed for {und}: {e}")
            und_rows.append({"underlying": _poly_symbol(und), "iv30": np.nan, "src": "none"})
        _sleep()

    und_df = pd.DataFrame(und_rows)
    logging.info(f"IV coverage — sources by underlying: { {k:v for k,v in coverage.items() if v>0} }")

    # Upsert today's Polygon-sourced IVs (skip carried_forward & none)
    to_upsert = []
    for _, r in und_df.iterrows():
        u = r["underlying"]; iv = r.get("iv30"); src = str(r.get("src",""))
        if pd.notna(iv) and src.startswith("polygon"):
            to_upsert.append({"date": today_iso, "underlying": u, "iv30": float(iv), "src": src})
    if to_upsert:
        upsert_ivhist(ivhist_path, today_iso, to_upsert)

    # Reload history after upsert (for consistent ranking)
    ivhist = load_ivhist(ivhist_path)

    # Output records
    records=[]
    for fund in tickers:
        grp = GROUP_OF[fund]; und = UNDERLYING.get(fund)
        ex_date  = ex_by_group[grp]; pay_date = pay_by_group[grp]; d2ex = (ex_date - today).days
        px = price_map.get(fund); px_asof = fmt_asof(asof_map.get(fund))

        # Div history & forecast
        ts, closes, divs = yahoo_chart_with_divs(session, fund, "18mo", "1d")
        events = build_events(ts, closes, divs)
        ex_ts = int(datetime(ex_date.year, ex_date.month, ex_date.day, 14, 30, tzinfo=timezone.utc).timestamp())
        prev_events = [e for e in events if e.ts < ex_ts]
        params = PARAMS_WEEKLY if is_weekly(fund) else PARAMS_MONTHLY
        y_hat = predict_hist_mom_simple(prev_events, ts, closes, ex_ts, **params)

        # If no fund history, fallback to per-period yield from IV
        iv_now = None
        if und:
            row = und_df[und_df["underlying"] == _poly_symbol(und)]
            if not row.empty and pd.notna(row.iloc[0]["iv30"]):
                iv_now = float(row.iloc[0]["iv30"])
        if y_hat is None and iv_now is not None:
            y_hat = per_period_yield_from_iv(iv_now, grp)

        ytp_point = (y_hat * px) if (y_hat is not None and px is not None) else None
        est_low = est_high = None
        if ytp_point is not None and px is not None and px > 0:
            half = abs(range_pct)
            est_low  = ytp_point * (1.0 - half)
            est_high = ytp_point * (1.0 + half)
            if est_low < 0: est_low = 0.0
            if est_high < est_low: est_high = est_low
            ytp_range_str = f"${est_low:.2f}-${est_high:.2f}"
            fwd_yield_ann = f"{((ytp_point/px)*FREQ_BY_GROUP[grp])*100:.1f}%"
            ytp_point_str = f"${ytp_point:.2f}"
        else:
            ytp_point_str=""; ytp_range_str=""; fwd_yield_ann=""

        # Historical payout range
        hist_low, hist_high, hist_n, hist_label = historical_dividend_range(events, grp, ex_ts)
        hist_range_str = f"${hist_low:.2f}-${hist_high:.2f}" if (hist_low is not None and hist_high is not None) else ""

        # RV30 for underlying (Yahoo)
        rv30 = None
        if und:
            ts_u, adj_u = yahoo_chart_hist_adj(session, und, rng="1y", interval="1d")
            vals = [a for a in adj_u if a is not None]
            if len(vals) >= 30:
                s = pd.Series(vals, dtype="float64")
                rv = float(np.log(s/s.shift(1)).dropna().tail(30).std(ddof=0) * np.sqrt(252))
                rv30 = rv if np.isfinite(rv) and rv>0 else None

        # Self IV Rank / Percentile
        rank_0_100 = None; pct_0_1 = None
        if und and (iv_now is not None or allow_bootstrap):
            # If iv_now missing but bootstrap allowed, try carry-forward for rank only
            if iv_now is None and allow_bootstrap:
                dfu = ivhist[ivhist["underlying"] == _poly_symbol(und)]
                if not dfu.empty:
                    dfu = dfu.sort_values("date")
                    last = dfu.iloc[-1]
                    try:
                        iv_cf = float(last["iv30"])
                        iv_now = iv_cf if 0.05 <= iv_cf <= 2.00 else None
                    except Exception:
                        iv_now = None
            if iv_now is not None:
                min_pts = 10 if allow_bootstrap else 30
                rank_0_100, pct_0_1 = compute_self_iv_percentiles(
                    ivhist, _poly_symbol(und), iv_now, lookback_days=365, min_points=min_pts,
                    polygon_only=True, ew_half_life_days=30.0
                )

        # IV–RV gap
        iv_rv_gap_pct = None
        if (iv_now is not None) and (rv30 is not None) and rv30 > 0:
            iv_rv_gap_pct = float(100.0 * (iv_now/rv30 - 1.0))
            # Log suspicious ratios
            ratio = iv_now/rv30
            if ratio < 0.4 or ratio > 4.0:
                logging.warning(f"IV/RV outlier for {und}: IV30={iv_now:.3f}, RV30={rv30:.3f}, ratio={ratio:.2f}")

        records.append({
            "fund": fund, "Group": grp, "Underlying": und or "",
            "Ex-Date": ex_date.isoformat(), "Pay Date": pay_date.isoformat(),
            "Price": (f"{px:.2f}" if (px is not None and np.isfinite(px)) else ""),
            "Price ($)": (float(px) if (px is not None and np.isfinite(px)) else np.nan),
            "Price as of": px_asof,
            "Est. Dividend": ytp_point_str,
            "Est. Dividend ($)": (float(ytp_point) if (ytp_point is not None and np.isfinite(ytp_point)) else np.nan),
            "Est. Range": ytp_range_str, "EstLow": (float(est_low) if est_low is not None else np.nan),
            "EstHigh": (float(est_high) if est_high is not None else np.nan),
            "Fwd Yield (Ann.)": fwd_yield_ann,
            "IV_Rank_Self_0_100": (float(rank_0_100) if rank_0_100 is not None else np.nan),
            "IV_Percentile_Self_0_1": (float(pct_0_1) if pct_0_1 is not None else np.nan),
            "RV30": (float(rv30) if rv30 is not None else np.nan),
            "IV-RV Gap (%)": (f"{iv_rv_gap_pct:.1f}%" if iv_rv_gap_pct is not None else ""),
            "IV_RV_Gap_Pct": (float(iv_rv_gap_pct) if iv_rv_gap_pct is not None else np.nan),
            "Hist Range (13m/52w)": hist_range_str, "HistLow": (float(hist_low) if hist_low is not None else np.nan),
            "HistHigh": (float(hist_high) if hist_high is not None else np.nan),
            "HistN": int(hist_n), "HistLabel": hist_label,
            "Days to Ex": d2ex if d2ex is not None else "",
        })

    # DataFrame
    df = pd.DataFrame.from_records(records)

    # Column order
    cols_front = [
        "fund","Group","Underlying","Ex-Date","Pay Date",
        "Price","Price ($)","Price as of",
        "Est. Dividend","Est. Dividend ($)","Est. Range","EstLow","EstHigh","Fwd Yield (Ann.)",
        "IV_Rank_Self_0_100","IV_Percentile_Self_0_1",
        "RV30","IV-RV Gap (%)","IV_RV_Gap_Pct",
        "Hist Range (13m/52w)","HistLow","HistHigh","HistN","HistLabel","Days to Ex"
    ]
    front = [c for c in cols_front if c in df.columns]
    rest  = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # Sort by Ex-Date
    def _to_dt(x):
        try: return pd.to_datetime(x).date()
        except Exception: return pd.NaT
    if "Ex-Date" in df.columns:
        df["_ex_sort"] = df["Ex-Date"].apply(_to_dt)
        df = df.sort_values("_ex_sort", na_position="last", kind="mergesort").drop(columns=["_ex_sort"]).reset_index(drop=True)

    # Save
    if save_path:
        if save_path.lower().endswith(".xlsx"):
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
        logging.info(f"Saved output → {save_path}")

    return df

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default="ytp_output.csv", help="CSV or XLSX output path")
    ap.add_argument("--range-pct", type=float, default=RANGE_PCT_DEFAULT, help="± half-width for Est. Range (default 0.10)")
    ap.add_argument("--polygon-key", type=str, default=None, help="Polygon API key (overrides env and file const)")
    ap.add_argument("--polygon-base-url", type=str, default="https://api.polygon.io", help="Polygon base URL")
    ap.add_argument("--allow-fallback-if-no-polygon", action="store_true", help="Run without Polygon (IV ranks become NaN unless bootstrapped)")
    ap.add_argument("--allow-bootstrap", action="store_true", help="If IV missing or history <30, relax min N to 10 and allow carry-forward IV for rank only.")
    ap.add_argument("--ivhist", type=str, default=IVHIST_PATH_DEFAULT, help="IV history CSV path for self-ranking")
    args = ap.parse_args()

    key, source = resolve_polygon_key(args)
    if key:
        logging.info(f"Using Polygon key from: {source}")

    df = run(save_path=args.save, range_pct=args.range_pct,
             polygon_key=key, polygon_base_url=args.polygon_base_url,
             ivhist_path=args.ivhist,
             allow_no_polygon=args.allow_fallback_if_no_polygon,
             allow_bootstrap=args.allow_bootstrap)

    with pd.option_context("display.max_rows", 500, "display.max_columns", None):
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()

