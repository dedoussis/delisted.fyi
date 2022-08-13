import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from time import sleep
import requests
import typing as t

from jinja2 import Environment, FileSystemLoader

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DailyQuote:
    def __init__(self, raw_data: t.Mapping[str, str]) -> None:
        self._raw_data = raw_data

    @property
    def close(self) -> float:
        return float(self._raw_data["4. close"])


class AlphaVantageClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.requester = requests.Session()
        self.requester.params["apikey"] = api_key  # type: ignore

    def get_time_series_daily(self, symbol: str) -> t.Mapping[datetime, DailyQuote]:
        logger.info("Querying daily time series for %s", symbol)
        resp = self.requester.get(
            f"{self.base_url}/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
            },
        ).json()
        ts_daily_raw: t.Mapping[str, t.Mapping[str, str]] = resp["Time Series (Daily)"]

        return {
            datetime.strptime(k, "%Y-%m-%d"): DailyQuote(raw_data=v)
            for k, v in ts_daily_raw.items()
        }


class DelistingData(t.NamedTuple):
    avg_closing_price: float
    days_until_delisting: int
    delisted: bool

    def to_json(self) -> t.Mapping[str, t.Any]:
        return {
            "avgClosingPrice": self.avg_closing_price,
            "daysUntilDelisting": self.days_until_delisting,
            "isDelisted": self.delisted,
        }


def calculate_days_until_delisting(
    curr_average_closing_price: float,
    consecutive_days_below_threshold: int,
    delisting_window_len: int,
    avg_closing_price_window_len: int,
    avg_closing_price_threshold: float,
    daily_rate: float,
    curr_sorted_average_closing_price_window: t.Sequence[float],
) -> int:
    if daily_rate < 0 and curr_average_closing_price >= avg_closing_price_threshold:
        days_until_below_threshold = 0
        curr_sorted_average_closing_price_window_q = deque(
            curr_sorted_average_closing_price_window
        )
        avg_closing_prices_sum = sum(curr_sorted_average_closing_price_window_q)

        while (
            avg_closing_prices_sum
            >= avg_closing_price_threshold * avg_closing_price_window_len
        ):
            avg_closing_prices_sum -= curr_sorted_average_closing_price_window_q.pop()
            curr_sorted_average_closing_price_window_q.appendleft(
                curr_sorted_average_closing_price_window_q[0] * (1 + daily_rate)
            )
            avg_closing_prices_sum += curr_sorted_average_closing_price_window_q[0]
            days_until_below_threshold += 1

        return delisting_window_len + days_until_below_threshold

    if daily_rate > 0 and curr_average_closing_price < avg_closing_price_threshold:
        for _ in range(delisting_window_len - consecutive_days_below_threshold):
            curr_average_closing_price *= 1 + daily_rate
            if curr_average_closing_price >= avg_closing_price_threshold:
                return -1

        return delisting_window_len - consecutive_days_below_threshold

    if daily_rate <= 0 and curr_average_closing_price < avg_closing_price_threshold:
        return delisting_window_len - consecutive_days_below_threshold

    # daily_rate >= 0 and curr_average_closing_price >= avg_closing_price_threshold
    return -1


def get_delisting_data(
    client: AlphaVantageClient,
    symbol: str,
    delisting_window_len: int,
    avg_closing_price_window_len: int,
    avg_closing_price_threshold: float,
) -> DelistingData:
    ts_daily = client.get_time_series_daily(symbol)
    sorted_closing_prices = [v.close for _, v in sorted(ts_daily.items(), reverse=True)]

    sorted_closing_prices_latest_window = sorted_closing_prices[
        :avg_closing_price_window_len
    ]
    avg_closing_price = (
        sum(sorted_closing_prices_latest_window) / avg_closing_price_window_len
    )

    avg_closing_prices = [avg_closing_price]
    for i in range(len(sorted_closing_prices) - avg_closing_price_window_len):
        avg_closing_prices.append(
            avg_closing_prices[-1]
            + (
                sorted_closing_prices[i + avg_closing_price_window_len]
                - sorted_closing_prices[i]
            )
            / avg_closing_price_window_len
        )

    consecutive_days_below_threshold = 0
    for avg_closing_price in avg_closing_prices:
        if avg_closing_price < avg_closing_price_threshold:
            consecutive_days_below_threshold += 1
        else:
            break

    daily_rate_sum: float = 0
    for i in range(1, len(sorted_closing_prices[:delisting_window_len])):
        current = sorted_closing_prices[i - 1]
        previous = sorted_closing_prices[i]

        daily_rate_sum += (current - previous) / previous

    daily_rate = daily_rate_sum / delisting_window_len

    return DelistingData(
        avg_closing_price=avg_closing_prices[0],
        days_until_delisting=calculate_days_until_delisting(
            curr_average_closing_price=avg_closing_prices[0],
            delisting_window_len=delisting_window_len,
            avg_closing_price_window_len=avg_closing_price_window_len,
            avg_closing_price_threshold=avg_closing_price_threshold,
            daily_rate=daily_rate,
            curr_sorted_average_closing_price_window=sorted_closing_prices_latest_window,
            consecutive_days_below_threshold=consecutive_days_below_threshold,
        ),
        delisted=consecutive_days_below_threshold > delisting_window_len,
    )


def get_delisting_data_for_symbols(
    client: AlphaVantageClient,
    symbols: t.Sequence[str],
    delisting_window_len: int,
    avg_closing_price_window_len: int,
    avg_closing_price_threshold: float,
    max_calls_per_min: int,
) -> t.Mapping[str, DelistingData]:
    dd = {}
    for i, symbol in enumerate(symbols):
        if i % max_calls_per_min == max_calls_per_min - 1:
            sleep(60)

        dd[symbol] = get_delisting_data(
            client,
            symbol,
            delisting_window_len,
            avg_closing_price_window_len,
            avg_closing_price_threshold,
        )

    return dd


def make_output(
    delisting_data: t.Mapping[str, DelistingData],
    template_dir: Path,
    output_dir: Path,
) -> None:
    json_data = json.dumps({k: v.to_json() for k, v in delisting_data.items()})
    updated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    env = Environment(loader=FileSystemLoader(template_dir))

    output_dir.mkdir(exist_ok=True)

    for template_name in env.list_templates():
        template = env.get_template(template_name)
        rendered_str = template.render(
            json_data=json_data,
            updated_at=updated_at,
        )

        output_file_path = output_dir / template_name.rstrip(".j2")
        with output_file_path.open("w+") as f:
            f.write(rendered_str)
        logger.info("Generated %s", output_file_path)


class Args(t.Protocol):
    symbols: t.Sequence[str]
    apikey: str
    base_url: str
    template_dir: Path
    output_dir: Path


class Namespace(t.Protocol):
    def parse_args(self) -> Args:
        ...


def build_parser() -> Namespace:
    parser = argparse.ArgumentParser(description="Delisted")
    parser.add_argument(
        "--symbols",
        nargs="+",
    )
    parser.add_argument(
        "--apikey",
        help="See https://www.alphavantage.co/support/#api-key",
    )
    parser.add_argument(
        "--base-url",
        help="AlphaVantage base URL",
        default="https://www.alphavantage.co",
    )
    parser.add_argument(
        "--template-dir",
        help="template directory",
        type=Path,
        default=Path.cwd() / "templates",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory",
        type=Path,
        default=Path.cwd() / "site",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = AlphaVantageClient(base_url=args.base_url, api_key=args.apikey)
    delisting_data = get_delisting_data_for_symbols(
        max_calls_per_min=5,
        client=client,
        symbols=args.symbols,
        delisting_window_len=30,
        avg_closing_price_window_len=5,
        avg_closing_price_threshold=1,
    )
    make_output(delisting_data, args.template_dir, args.output_dir)


if __name__ == "__main__":
    main()
