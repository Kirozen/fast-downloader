from __future__ import annotations

import multiprocessing
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from requests.models import HTTPError
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@dataclass
class DownloadFile:
    urls: list[str]
    dest: Path = Path.cwd()
    filename: str = field(init=False)

    def __post_init__(self):
        self.filename = Path(self.urls[0]).name

    @property
    def filepath(self):
        return self.dest / self.filename


BUFFER_SIZE = 32768

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


def parse_aria2(data: list[str], destination: Path):
    files = []
    out_re = re.compile(r"^\s+out=(?P<out>.*)$")
    for line in data:
        if line.startswith("#") or not line:
            continue
        if line.startswith("http"):
            files.append(DownloadFile(line.split("\t"), destination))
        else:
            match_out = out_re.match(line)
            if match_out:
                files[-1].filename = match_out.groupdict()["out"]
    return files


def get_inputs(inputs: list[str], destination: Path, aria2_compatibility: bool):
    paths = []
    for input in inputs:
        lines = Path(input).read_text().splitlines(keepends=False)
        if aria2_compatibility:
            paths.extend(parse_aria2(lines, destination))
        else:
            paths.extend(
                DownloadFile([url], destination)
                for url in lines
                if url.startswith("http")
            )
    return paths


def downloader(downloadfile: DownloadFile, buffer_size: int, quiet: bool):
    if not quiet:
        task_id = progress.add_task(
            "download",
            filename=downloadfile.filename,
        )
    iterator = iter(downloadfile.urls)
    response = None
    try:
        while not response:
            url = next(iterator)
            try:
                response = requests.get(url, allow_redirects=True, stream=True)
                response.raise_for_status()
            except HTTPError:
                response = None
        if not quiet:
            size = int(response.headers.get("content-length"))
            progress.update(task_id, total=size)
        with open(downloadfile.filepath, "wb") as handler:
            if not quiet:
                progress.start_task(task_id)
            for data in response.iter_content(chunk_size=buffer_size):
                handler.write(data)
                if not quiet:
                    progress.update(task_id, advance=len(data))
    except StopIteration:
        print("Urls are not available")


def executor(threads, downloadfiles, buffer_size, quiet):
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for downloadfile in sorted(
            downloadfiles, key=lambda df: len(df.filename), reverse=True
        ):
            try:
                for url in downloadfile.urls:
                    urlparse(url)
            except ValueError:
                print(f"An url in {downloadfile.urls} is not valid!", file=sys.stderr)
                continue
            pool.submit(downloader, downloadfile, buffer_size, quiet)


@click.command()
@click.option(
    "-t",
    "--threads",
    default=lambda: multiprocessing.cpu_count(),
    type=click.IntRange(min=1, max=1000, clamp=True),
    help="thread number",
)
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    type=click.Path(exists=True, file_okay=True),
    help="input file",
)
@click.option("-q", "--quiet", is_flag=True)
@click.option(
    "-d",
    "--destination",
    type=click.Path(dir_okay=True, allow_dash=True),
    default=Path(os.getcwd()),
)
@click.option("--aria2-compatibility", is_flag=True)
@click.option(
    "--buffer-size", type=click.IntRange(min=1, clamp=True), default=BUFFER_SIZE
)
@click.argument("urls", nargs=-1, type=click.Path())
def fast_downloader(
    threads, inputs, quiet, destination, buffer_size, aria2_compatibility, urls
):
    download_urls = (DownloadFile([url], Path(destination)) for url in urls)
    download_files = list(
        chain(download_urls, get_inputs(inputs, Path(destination), aria2_compatibility))
    )

    if quiet:
        executor(threads, download_files, buffer_size, quiet)
    else:
        with progress:
            executor(threads, download_files, buffer_size, quiet)


if __name__ == "__main__":
    fast_downloader()
