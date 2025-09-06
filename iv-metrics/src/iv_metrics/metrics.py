from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _to_series(values: Iterable[float] | pd.Series) -> pd.Series:
	if isinstance(values, pd.Series):
		series = values.copy()
	else:
		series = pd.Series(list(values))
	return series.astype(float)


def _last_window(series: pd.Series, window: int) -> pd.Series:
	if window <= 0:
		raise ValueError("window must be positive")
	if len(series) == 0:
		raise ValueError("series is empty")
	if len(series) < window:
		# Use full history if shorter than window
		return series.dropna()
	return series.iloc[-window:].dropna()


def iv_rank(values: Iterable[float] | pd.Series, window: int = 252) -> float:
	"""Compute IV Rank over the lookback window.

	Definition (common):
	  IV Rank = 100 * (IV_last - IV_min) / (IV_max - IV_min)

	Returns NaN if window contains fewer than 2 non-NaN points or zero range.
	"""
	series = _to_series(values)
	window_series = _last_window(series, window)

	if window_series.size < 2:
		return float("nan")

	iv_min = float(window_series.min())
	iv_max = float(window_series.max())
	iv_last = float(window_series.iloc[-1])

	if math.isclose(iv_max, iv_min, rel_tol=0.0, abs_tol=0.0):
		return float("nan")

	return 100.0 * (iv_last - iv_min) / (iv_max - iv_min)


def iv_percentile(values: Iterable[float] | pd.Series, window: int = 252) -> float:
	"""Compute IV Percentile of the last value relative to the window.

	Percentile definition used:
	  share of values in window that are <= last value (including ties).

	Returns NaN if window contains fewer than 1 non-NaN point.
	"""
	series = _to_series(values)
	window_series = _last_window(series, window)

	if window_series.size < 1:
		return float("nan")

	last_value = float(window_series.iloc[-1])
	# Count less-or-equal values including last
	count_le = int((window_series <= last_value).sum())
	n = int(window_series.size)

	return 100.0 * count_le / n
