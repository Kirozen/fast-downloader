from __future__ import annotations

import multiprocessing
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import click
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests.models import HTTPError
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class AuthMethod(Enum):
    NoAuth = 0
    Basic = 1
    Digest = 2

    @classmethod
    def from_string(cls, s: str):
        if s.lower() == "basic":
            return cls.Basic
        elif s.lower() == "digest":
            return cls.Digest
        else:
            return cls.NoAuth


@dataclass
class DownloadFile:
    urls: list[str]
    dest: Path = Path.cwd()
    auth_method: AuthMethod = AuthMethod.NoAuth
    user: str = ""
    password: str = ""
    filename: str = field(init=False)

    def __post_init__(self):
        self.filename = Path(self.urls[0]).name

    @property
    def filepath(self):
        return self.dest / self.filename

    def get_auth(self):
        if self.auth_method == AuthMethod.Basic:
            return HTTPBasicAuth(self.user, self.password)
        elif self.auth_method == AuthMethod.Digest:
            return HTTPDigestAuth(self.user, self.password)
        return None


@dataclass
class DownloadOptions:
    threads: int
    quiet: bool
    buffer_size: int


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
    out_re = re.compile(r"^\s+out=(?P<out>.*)$")
    user_re = re.compile(r"^\s+http-user=(?P<user>.*)$")
    password_re = re.compile(r"^\s+http-passwd=(?P<password>.*)$")

    files = []
    for line in data:
        if line.startswith("#") or not line:
            continue
        if line.startswith("http"):
            files.append(DownloadFile(line.split("\t"), destination))
            continue
        match_out = out_re.match(line)
        if match_out:
            files[-1].filename = match_out.groupdict()["out"]
            continue
        match_user = user_re.match(line)
        if match_user:
            files[-1].user = match_user.groupdict()["user"]
            continue
        match_password = password_re.match(line)
        if match_password:
            files[-1].password = match_password.groupdict()["password"]

    for file in files:
        if file.user and file.password:
            file.auth_method = AuthMethod.Basic

    return files


def get_inputs(
    inputs: list[str],
    destination: Path,
    aria2_compatibility: bool,
    method: AuthMethod,
    user: str,
    password: str,
):
    paths = []
    for input in inputs:
        lines = Path(input).read_text().splitlines(keepends=False)
        if aria2_compatibility:
            paths.extend(parse_aria2(lines, destination))
        else:
            paths.extend(
                DownloadFile([url], destination, method, user, password)
                for url in lines
                if url.startswith("http")
            )
    return paths


def downloader(downloadfile: DownloadFile, download_options: DownloadOptions):
    if not download_options.quiet:
        task_id = progress.add_task(
            "download",
            filename=downloadfile.filename,
        )
    buffer_size = download_options.buffer_size
    auth = downloadfile.get_auth()
    iterator = iter(downloadfile.urls)

    try:
        response = None
        while not response:
            url = next(iterator)
            try:
                response = requests.get(
                    url, allow_redirects=True, stream=True, auth=auth
                )
                response.raise_for_status()
            except HTTPError:
                response = None

        if not download_options.quiet:
            size = int(response.headers.get("content-length"))
            progress.update(task_id, total=size)
        with open(downloadfile.filepath, "wb") as handler:
            if not download_options.quiet:
                progress.start_task(task_id)
            for data in response.iter_content(chunk_size=buffer_size):
                handler.write(data)
                if not download_options.quiet:
                    progress.update(task_id, advance=len(data))
    except StopIteration:
        print("Urls are not available", file=sys.stderr)


def executor(downloadfiles: list[DownloadFile], download_options: DownloadOptions):
    with ThreadPoolExecutor(max_workers=download_options.threads) as pool:
        for downloadfile in sorted(
            downloadfiles, key=lambda df: len(df.filename), reverse=True
        ):
            try:
                for url in downloadfile.urls:
                    urlparse(url)
            except ValueError:
                print(f"An url in {downloadfile.urls} is not valid!", file=sys.stderr)
                continue
            pool.submit(downloader, downloadfile, download_options)


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
    help="destination folder",
)
@click.option("--aria2-compatibility", is_flag=True)
@click.option(
    "-b",
    "--buffer-size",
    type=click.IntRange(min=1, clamp=True),
    default=BUFFER_SIZE,
    help="buffer size",
)
@click.option(
    "--auth-method",
    type=click.Choice(["basic", "digest"], case_sensitive=False),
    default="noauth",
)
@click.option("-u", "--user", type=str, help="user for authentication")
@click.option("-p", "--password", type=str, help="password for authentication")
@click.argument("urls", nargs=-1, type=click.Path())
def fast_downloader(
    threads: int,
    inputs: list[str],
    quiet: bool,
    destination: str,
    buffer_size: int,
    aria2_compatibility: bool,
    urls: list[str],
    auth_method: str,
    user: str,
    password: str,
):
    download_options = DownloadOptions(threads, quiet, buffer_size)
    method = AuthMethod.from_string(auth_method)
    if method == AuthMethod.NoAuth and user and password:
        method = AuthMethod.Basic
    dest_path = Path(destination)

    download_urls = (
        DownloadFile([url], dest_path, method, user, password) for url in urls
    )
    urls_from_inputs = get_inputs(
        inputs, dest_path, aria2_compatibility, method, user, password
    )
    download_files = list(chain(download_urls, urls_from_inputs))

    if quiet:
        executor(download_files, download_options)
    else:
        with progress:
            executor(download_files, download_options)


if __name__ == "__main__":
    fast_downloader()
