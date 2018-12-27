#!/usr/bin/env python

import sys
import json

crates = json.loads(sys.stdin.readline().strip())["prelude"]["external_crates"]
assert any(map(lambda c: c["id"]["name"] == "krate2", crates))
