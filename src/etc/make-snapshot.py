#!/usr/bin/env python

import snapshot, sys

if len(sys.argv) == 3:
    print(snapshot.make_snapshot(sys.argv[1], sys.argv[2], ""))
else:
    print(snapshot.make_snapshot(sys.argv[1], sys.argv[2], sys.argv[3]))
