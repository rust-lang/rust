#!/usr/bin/env python

import sys
import json
import os.path

for line in sys.stdin:
    json_data = json.loads(line)
    if "stdout" in json_data and "f.rs" in json_data["stdout"]:
        normalized_path = os.path.join(os.getcwd(), "f.rs")
        json_data["stdout"] = json_data["stdout"].replace("f.rs", normalized_path)
    result = json.dumps(json_data).replace("{", "{ ")
    result = result.replace("}", " }")
    print(result)
