#!/usr/bin/env python

import sys
import json

# Try to decode line in order to ensure it is a valid JSON document
for line in sys.stdin:
    parsed = json.loads(line)
    print (json.dumps(parsed, indent=2, separators=(',', ': '), sort_keys=True))
