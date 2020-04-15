#!/usr/bin/env python

import json
import os
import sys

from lintlib import log


def key(v):
    if v == 'master':
        return float('inf')
    if v == 'stable':
        return sys.maxsize
    if v == 'beta':
        return sys.maxsize - 1

    v = v.replace('v', '').replace('rust-', '')

    s = 0
    for i, val in enumerate(v.split('.')[::-1]):
        s += int(val) * 100**i

    return s


def main():
    if len(sys.argv) < 2:
        print("Error: specify output directory")
        return

    outdir = sys.argv[1]
    versions = [
        dir for dir in os.listdir(outdir) if not dir.startswith(".") and os.path.isdir(os.path.join(outdir, dir))
    ]
    versions.sort(key=key)

    with open(os.path.join(outdir, "versions.json"), "w") as fp:
        json.dump(versions, fp, indent=2)
        log.info("wrote JSON for great justice")


if __name__ == "__main__":
    main()
