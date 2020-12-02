#!/usr/bin/env python

# This script can check that an expected json blob is a subset of what actually gets produced.
# The comparison is independent of the value of IDs (which are unstable) and instead uses their
# relative ordering to check them against eachother by looking them up in their respective blob's
# `index` or `paths` mappings. To add a new test run `rustdoc --output-format json -o . yourtest.rs`
# and then create `yourtest.expected` by stripping unnecessary details from `yourtest.json`. If
# you're on windows, replace `\` with `/`.

import copy
import sys
import json
import types

# Used instead of the string ids when used as references.
# Not used as keys in `index` or `paths`
class ID(str):
    pass


class SubsetException(Exception):
    def __init__(self, msg, trace):
        self.msg = msg
        self.trace = msg
        super().__init__("{}: {}".format(trace, msg))


def check_subset(expected_main, actual_main, base_dir):
    expected_index = expected_main["index"]
    expected_paths = expected_main["paths"]
    actual_index = actual_main["index"]
    actual_paths = actual_main["paths"]
    already_checked = set()

    def _check_subset(expected, actual, trace):
        expected_type = type(expected)
        actual_type = type(actual)

        if actual_type is str:
            actual = normalize(actual).replace(base_dir, "$TEST_BASE_DIR")

        if expected_type is not actual_type:
            raise SubsetException(
                "expected type `{}`, got `{}`".format(expected_type, actual_type), trace
            )


        if expected_type in (int, bool, str) and expected != actual:
            raise SubsetException("expected `{}`, got: `{}`".format(expected, actual), trace)
        if expected_type is dict:
            for key in expected:
                if key not in actual:
                    raise SubsetException(
                        "Key `{}` not found in output".format(key), trace
                    )
                new_trace = copy.deepcopy(trace)
                new_trace.append(key)
                _check_subset(expected[key], actual[key], new_trace)
        elif expected_type is list:
            expected_elements = len(expected)
            actual_elements = len(actual)
            if expected_elements != actual_elements:
                raise SubsetException(
                    "Found {} items, expected {}".format(
                        expected_elements, actual_elements
                    ),
                    trace,
                )
            for expected, actual in zip(expected, actual):
                new_trace = copy.deepcopy(trace)
                new_trace.append(expected)
                _check_subset(expected, actual, new_trace)
        elif expected_type is ID and expected not in already_checked:
            already_checked.add(expected)
            _check_subset(
                expected_index.get(expected, {}), actual_index.get(actual, {}), trace
            )
            _check_subset(
                expected_paths.get(expected, {}), actual_paths.get(actual, {}), trace
            )

    _check_subset(expected_main["root"], actual_main["root"], [])


def rustdoc_object_hook(obj):
    # No need to convert paths, index and external_crates keys to ids, since
    # they are the target of resolution, and never a source itself.
    if "id" in obj and obj["id"]:
        obj["id"] = ID(obj["id"])
    if "root" in obj:
        obj["root"] = ID(obj["root"])
    if "items" in obj:
        obj["items"] = [ID(id) for id in obj["items"]]
    if "variants" in obj:
        obj["variants"] = [ID(id) for id in obj["variants"]]
    if "fields" in obj:
        obj["fields"] = [ID(id) for id in obj["fields"]]
    if "impls" in obj:
        obj["impls"] = [ID(id) for id in obj["impls"]]
    if "implementors" in obj:
        obj["implementors"] = [ID(id) for id in obj["implementors"]]
    if "links" in obj:
        obj["links"] = {s: ID(id) for s, id in obj["links"]}
    if "variant_kind" in obj and obj["variant_kind"] == "struct":
        obj["variant_inner"] = [ID(id) for id in obj["variant_inner"]]
    return obj


def main(expected_fpath, actual_fpath, base_dir):
    print(
        "checking that {} is a logical subset of {}".format(
            expected_fpath, actual_fpath
        )
    )
    with open(expected_fpath) as expected_file:
        expected_main = json.load(expected_file, object_hook=rustdoc_object_hook)
    with open(actual_fpath) as actual_file:
        actual_main = json.load(actual_file, object_hook=rustdoc_object_hook)
    check_subset(expected_main, actual_main, base_dir)
    print("all checks passed")

def normalize(s):
    return s.replace('\\', '/')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: `compare.py expected.json actual.json test-dir`")
    else:
        main(sys.argv[1], sys.argv[2], normalize(sys.argv[3]))
