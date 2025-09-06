import argparse
import math
import sys

import pandas as pd

from .metrics import iv_rank, iv_percentile


def _fmt(value: float) -> str:
	return "NaN" if (value is None or (isinstance(value, float) and math.isnan(value))) else f"{value:.2f}"


def main() -> None:
	parser = argparse.ArgumentParser(description="Compute IV metrics from CSV.")
	parser.add_argument("csv", help="Path to CSV with date and iv columns")
	parser.add_argument("--date-column", default="date", help="Name of date column")
	parser.add_argument("--iv-column", default="iv", help="Name of implied volatility column")
	parser.add_argument("--window", type=int, default=252, help="Lookback window length")
	parser.add_argument("--as-of", help="As-of date (YYYY-MM-DD). Defaults to last row.")

	args = parser.parse_args()

	try:
		df = pd.read_csv(args.csv)
	except Exception as exc:
		print(f"Failed to read CSV: {exc}", file=sys.stderr)
		sys.exit(1)

	if args.date_column in df.columns:
		df[args.date_column] = pd.to_datetime(df[args.date_column])
		df = df.sort_values(args.date_column)
	else:
		df = df.reset_index(drop=True)

	if args.iv_column not in df.columns:
		print(f"Missing IV column: {args.iv_column}", file=sys.stderr)
		sys.exit(1)

	iv_series = df[args.iv_column].astype(float)

	if args.as_of is not None:
		as_of = pd.to_datetime(args.as_of)
		if args.date_column in df.columns:
			mask = df[args.date_column] <= as_of
			iv_series = df.loc[mask, args.iv_column].astype(float)

	rank = iv_rank(iv_series, window=args.window)
	pct = iv_percentile(iv_series, window=args.window)

	print(f"IV Rank: {_fmt(rank)}")
	print(f"IV Percentile: {_fmt(pct)}")
