# Fast Downloader

## Why?

Primarly, to have a nice tool to download ubuntu packages with apt-fast.

## Documents

#### [CHANGELOG.md](https://github.com/kirozen/fast-downloader/blob/master/CHANGELOG.md)
#### [LICENSE MIT](https://github.com/kirozen/fast-downloader/blob/master/LICENSE)


## Features

- Download multiple files in parallel
- Partially compatible with aria2 input file format

## Installing

Install with `pip` or your favorite PyPi package manager.

```shell
pip install fast-downloader-mt
```

## Dependencies

- Python 3.8+
- Rich
- Requests
- Click

## Usage

Usage: fdl [OPTIONS] [URLS]...

Options:
  -t, --threads INTEGER RANGE     thread number
  -i, --input PATH                input file
  -q, --quiet
  -d, --destination PATH          destination folder
  --aria2-compatibility
  -b, --buffer-size INTEGER RANGE
                                  buffer size
  --auth-method [basic|digest]
  -u, --user TEXT                 user for authentication
  -p, --password TEXT             password for authentication
  --help                          Show this message and exit.

Default number of threads set to `cpu_count()`

### Usage with apt-fast

In `/etc/apt-fast.conf`

```conf
_DOWNLOADER='fdl --aria2-compatibility -t ${_MAXNUM} -i ${DLLIST} -d ${DLDIR}'
```

## Dev environment

### Dev dependencies

- pytest
- pylint
- black
- mypy
- flake8
- bandit
- safety

## Setup

With poetry:

```shell
poetry install
```
