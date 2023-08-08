#!/usr/bin/env python

import os
import sys
import json


def find_redirect_map_file(folder, errors):
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith("redirect-map.json"):
                continue
            with open(os.path.join(root, name)) as f:
                data = json.load(f)
            with open("expected.json") as f:
                expected = json.load(f)
            for key in expected:
                if expected[key] != data.get(key):
                    errors.append("Expected `{}` for key `{}`, found: `{}`".format(
                        expected[key], key, data.get(key)))
                else:
                    del data[key]
            for key in data:
                errors.append("Extra data not expected: key: `{}`, data: `{}`".format(
                    key, data[key]))
            return True
    return False


if len(sys.argv) != 2:
    print("Expected doc directory to check!")
    sys.exit(1)

errors = []
if not find_redirect_map_file(sys.argv[1], errors):
    print("Didn't find the map file in `{}`...".format(sys.argv[1]))
    sys.exit(1)
for err in errors:
    print("=> {}".format(err))
if len(errors) != 0:
    sys.exit(1)
