#!/usr/bin/env python3

# `llvm-cov show` prints grouped subviews (e.g. for generic functions) in an
# unstable order, which is inconvenient when checking output snapshots with
# `diff`. To work around that, this script detects consecutive subviews in its
# piped input, and sorts them while preserving their contents.

from __future__ import print_function

import sys


def main():
    subviews = []

    def flush_subviews():
        if not subviews:
            return

        # The last "subview" should be just a boundary line on its own, so
        # temporarily remove it before sorting the accumulated subviews.
        terminator = subviews.pop()
        subviews.sort()
        subviews.append(terminator)

        for view in subviews:
            for line in view:
                print(line, end="")

        subviews.clear()

    for line in sys.stdin:
        if line.startswith("  ------------------"):
            # This is a subview boundary line, so start a new subview.
            subviews.append([line])
        elif line.startswith("  |"):
            # Add this line to the current subview.
            subviews[-1].append(line)
        else:
            # This line is not part of a subview, so sort and print any
            # accumulated subviews, and then print the line as-is.
            flush_subviews()
            print(line, end="")

    flush_subviews()
    assert not subviews


if __name__ == "__main__":
    main()
