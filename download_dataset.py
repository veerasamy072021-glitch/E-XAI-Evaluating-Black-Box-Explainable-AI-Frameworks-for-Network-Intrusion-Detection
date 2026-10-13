from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from modules.config import RAW_DIR, ensure_directories


def download(url: str, output: Path) -> Path:
    ensure_directories()
    output = output.resolve()
    raw_root = RAW_DIR.resolve()
    if raw_root not in output.parents:
        raise ValueError(f"Output must be inside {raw_root}")
    request = Request(url, headers={"User-Agent": "E-XAI/1.0"})
    with urlopen(request, timeout=60) as response, output.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a direct CSV URL into data/raw.")
    parser.add_argument("url", nargs="?", help="Direct URL to an authorized CICIDS2017/UNSW-NB15 CSV")
    parser.add_argument("--name", default="network_intrusion.csv", help="Destination filename")
    args = parser.parse_args()
    if not args.url:
        print("Place licensed CICIDS2017 or UNSW-NB15 CSV files in data/raw/, or pass a direct CSV URL.")
        print("Example: python download_dataset.py https://host/path/data.csv --name unsw_nb15.csv")
        return
    name = Path(args.name).name
    if not name.lower().endswith(".csv"):
        raise ValueError("The destination must be a .csv file.")
    print(f"Downloaded to {download(args.url, RAW_DIR / name)}")


if __name__ == "__main__":
    main()
