import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
from time import sleep
import requests
import typing as t
import operator
from jinja2 import Environment, FileSystemLoader

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AVG_CLOSING_PRICE_THRESHOLD = 1
TRADING_WINDOW_LEN = 30


class DailyQuote:
    def __init__(self, raw_data: t.Mapping[str, str]) -> None:
        self._raw_data = raw_data

    @property
    def close(self) -> float:
        return float(self._raw_data["4. close"])


class AlphaVantageClient:
    MAX_CALLS_PER_MIN = 5

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
    days_until_state_transition: int
    delisted: bool

    def to_json(self) -> t.Mapping[str, t.Any]:
        return {
            "avgClosingPrice": self.avg_closing_price,
            "daysUntilStateTransition": self.days_until_state_transition,
            "isDelisted": self.delisted,
        }


def calculate_days_until_state_transition(
    current_avg_closing_price: float,
    daily_rate: float,
    current_sorted_closing_prices_window: t.Sequence[float],
) -> int:
    def calculate(cmp: t.Callable[[float, float], bool]) -> int:
        days = 0
        curr_sorted_closing_price_window_q = deque(current_sorted_closing_prices_window)
        avg_closing_prices_sum = current_avg_closing_price * TRADING_WINDOW_LEN

        while cmp(
            avg_closing_prices_sum, AVG_CLOSING_PRICE_THRESHOLD * TRADING_WINDOW_LEN
        ):
            avg_closing_prices_sum -= curr_sorted_closing_price_window_q.pop()
            curr_sorted_closing_price_window_q.appendleft(
                curr_sorted_closing_price_window_q[0] * (1 + daily_rate)
            )
            avg_closing_prices_sum += curr_sorted_closing_price_window_q[0]
            days += 1

        return days

    if current_avg_closing_price >= AVG_CLOSING_PRICE_THRESHOLD:
        if daily_rate < 0:
            return calculate(operator.ge)

        return -1

    # average closing price below threshold:
    if daily_rate > 0:
        return calculate(operator.lt)

    # average closing price below threshold and negative daily rate:
    return -1


def get_delisting_data(
    client: AlphaVantageClient,
    symbol: str,
) -> DelistingData:
    ts_daily = client.get_time_series_daily(symbol)
    sorted_closing_prices = [v.close for _, v in sorted(ts_daily.items(), reverse=True)]
    sorted_closing_prices_latest_window = sorted_closing_prices[:TRADING_WINDOW_LEN]
    avg_closing_price = sum(sorted_closing_prices_latest_window) / TRADING_WINDOW_LEN

    # Calculate average daily rate of price change over the last 30 days
    daily_rate_sum: float = 0
    for i in range(1, len(sorted_closing_prices_latest_window)):
        current = sorted_closing_prices[i - 1]
        previous = sorted_closing_prices[i]

        daily_rate_sum += (current - previous) / previous

    daily_rate = daily_rate_sum / TRADING_WINDOW_LEN

    return DelistingData(
        avg_closing_price=avg_closing_price,
        days_until_state_transition=calculate_days_until_state_transition(
            current_avg_closing_price=avg_closing_price,
            daily_rate=daily_rate,
            current_sorted_closing_prices_window=sorted_closing_prices_latest_window,
        ),
        delisted=avg_closing_price < AVG_CLOSING_PRICE_THRESHOLD,
    )


def get_delisting_data_for_symbols(
    client: AlphaVantageClient,
    symbols: t.Sequence[str],
) -> t.Mapping[str, DelistingData]:
    dd = {}
    for i, symbol in enumerate(symbols):
        # Some hacky backpressure so that the rate limit is not reached
        if i % client.MAX_CALLS_PER_MIN == client.MAX_CALLS_PER_MIN - 1:
            sleep(60)

        dd[symbol] = get_delisting_data(
            client,
            symbol,
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
        client=client,
        symbols=args.symbols,
    )
    make_output(delisting_data, args.template_dir, args.output_dir)


if __name__ == "__main__":
    main()
