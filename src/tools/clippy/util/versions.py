#!/usr/bin/env python

import json
import logging as log
import os
import sys

log.basicConfig(level=log.INFO, format="%(levelname)s: %(message)s")


def key(v):
    if v == "master":
        return float("inf")
    if v == "stable":
        return sys.maxsize
    if v == "beta":
        return sys.maxsize - 1
    if v == "pre-1.29.0":
        return -1

    v = v.replace("rust-", "")

    s = 0
    for i, val in enumerate(v.split(".")[::-1]):
        s += int(val) * 100**i

    return s


def main():
    if len(sys.argv) < 2:
        log.error("specify output directory")
        return

    outdir = sys.argv[1]
    versions = [
        dir
        for dir in os.listdir(outdir)
        if not dir.startswith(".")
        and not dir.startswith("v")
        and os.path.isdir(os.path.join(outdir, dir))
    ]
    versions.sort(key=key)

    with open(os.path.join(outdir, "versions.json"), "w") as fp:
        json.dump(versions, fp, indent=2)
        log.info("wrote JSON for great justice")


if __name__ == "__main__":
    main()
