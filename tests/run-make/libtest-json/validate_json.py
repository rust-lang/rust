#!/usr/bin/env python

import sys
import json

# Try to decode line in order to ensure it is a valid JSON document
for line in sys.stdin:
    json.loads(line)
