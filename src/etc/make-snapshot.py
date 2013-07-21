#!/usr/bin/env python
# xfail-license

import snapshot, sys

print(snapshot.make_snapshot(sys.argv[1], sys.argv[2]))
