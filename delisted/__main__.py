import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import requests
import typing as t

from jinja2 import Environment, FileSystemLoader


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
        self.requester.params["apikey"] = api_key

    def get_time_series_daily(self, symbol: str) -> t.Mapping[datetime, DailyQuote]:
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
    closing_avg: float
    days: int
    delisted: bool


def get_delisting_data(
    client: AlphaVantageClient,
    symbol: str,
    window_len: int,
    closing_avg_threshold: float,
) -> DelistingData:
    ts_daily = client.get_time_series_daily(symbol)
    sorted_closes = [v.close for _, v in sorted(ts_daily.items(), reverse=True)]

    sorted_closes_window = deque(sorted_closes[:window_len])

    closing_avg = sum(sorted_closes_window) / window_len

    daily_rate_sum = 0
    for i in range(1, len(sorted_closes_window)):
        current = sorted_closes_window[i - 1]
        previous = sorted_closes_window[i]

        daily_rate_sum += (current - previous) / previous

    daily_rate = daily_rate_sum / window_len
    days = -1
    if daily_rate < 0:
        days = 0
        window_closing_sum = sum(sorted_closes_window)
        while window_closing_sum >= closing_avg_threshold * window_len:
            window_closing_sum -= sorted_closes_window.pop()
            sorted_closes_window.appendleft(
                sorted_closes_window[0] + (sorted_closes_window[0] * daily_rate)
            )
            window_closing_sum += sorted_closes_window[0]
            days += 1

    return DelistingData(
        closing_avg=closing_avg,
        days=days,
        delisted=closing_avg < closing_avg_threshold,
    )


def make_output(
    symbol: str, delisting_data: DelistingData, template_dir: Path, output_dir: Path
) -> None:
    env = Environment(loader=FileSystemLoader(template_dir))

    output_dir.mkdir(exist_ok=True)

    for template_name in env.list_templates():
        template = env.get_template(template_name)
        rendered_str = template.render(
            symbol=symbol,
            delisting_data=delisting_data,
            updated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        )

        output_file_path = output_dir / template_name.rstrip(".j2")
        with output_file_path.open("w+") as f:
            f.write(rendered_str)


class Args(t.Protocol):
    symbol: str
    apikey: str
    base_url: str
    template_dir: Path
    output_dir: Path


class Namespace(t.Protocol):
    def parse_args() -> Args:
        ...


def build_parser() -> Namespace:
    parser = argparse.ArgumentParser(description="Delisted")
    parser.add_argument(
        "--symbol",
        help="Stock symbol (ticker)",
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


def main():
    args = build_parser().parse_args()
    client = AlphaVantageClient(base_url=args.base_url, api_key=args.apikey)
    delisting_data = get_delisting_data(
        client=client, symbol=args.symbol, window_len=30, closing_avg_threshold=1
    )
    make_output(args.symbol, delisting_data, args.template_dir, args.output_dir)


if __name__ == "__main__":
    main()
