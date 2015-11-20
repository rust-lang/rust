# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This script does a tree-wide sanity checks against stability
# attributes, currently:
#   * For all feature_name/level pairs the 'since' field is the same
#   * That no features are both stable and unstable.
#   * That lib features don't have the same name as lang features
#     unless they are on the 'joint_features' whitelist
#   * That features that exist in both lang and lib and are stable
#     since the same version
#   * Prints information about features

import sys
import os
import re
import codecs

if len(sys.argv) < 2:
    print("usage: featureck.py <src-dir>")
    sys.exit(1)

src_dir = sys.argv[1]

# Features that are allowed to exist in both the language and the library
joint_features = [ ]

# Grab the list of language features from the compiler
language_gate_statuses = [ "Active", "Deprecated", "Removed", "Accepted" ]
feature_gate_source = os.path.join(src_dir, "libsyntax", "feature_gate.rs")
language_features = []
language_feature_names = []
with open(feature_gate_source, 'r') as f:
    for line in f:
        original_line = line
        line = line.strip()
        is_feature_line = False
        for status in language_gate_statuses:
            if status in line and line.startswith("("):
                is_feature_line = True

        if is_feature_line:
            # turn `    ("foo", "1.0.0", Some(10), Active)` into
            # `"foo", "1.0.0", Some(10), Active`
            line = line.strip(' ,()')
            parts = line.split(",")
            if len(parts) != 4:
                print("error: unexpected number of components in line: " + original_line)
                sys.exit(1)
            feature_name = parts[0].strip().replace('"', "")
            since = parts[1].strip().replace('"', "")
            issue = parts[2].strip()
            status = parts[3].strip()
            assert len(feature_name) > 0
            assert len(since) > 0
            assert len(issue) > 0
            assert len(status) > 0

            language_feature_names += [feature_name]
            language_features += [(feature_name, since, issue, status)]

assert len(language_features) > 0

errors = False

lib_features = { }
lib_features_and_level = { }
for (dirpath, dirnames, filenames) in os.walk(src_dir):
    # Don't look for feature names in tests
    if "src/test" in dirpath:
        continue

    # Takes a long time to traverse LLVM
    if "src/llvm" in dirpath:
        continue

    for filename in filenames:
        if not filename.endswith(".rs"):
            continue

        path = os.path.join(dirpath, filename)
        with codecs.open(filename=path, mode='r', encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                level = None
                if "[unstable(" in line:
                    level = "unstable"
                elif "[stable(" in line:
                    level = "stable"
                else:
                    continue

                # This is a stability attribute. For the purposes of this
                # script we expect both the 'feature' and 'since' attributes on
                # the same line, e.g.
                # `#[unstable(feature = "foo", since = "1.0.0")]`

                p = re.compile('(unstable|stable).*feature *= *"(\w*)"')
                m = p.search(line)
                if not m is None:
                    feature_name = m.group(2)
                    since = None
                    if re.compile("\[ *stable").search(line) is not None:
                        pp = re.compile('since *= *"([\w\.]*)"')
                        mm = pp.search(line)
                        if not mm is None:
                            since = mm.group(1)
                        else:
                            print("error: misformed stability attribute")
                            print("line %d of %:" % (line_num, path))
                            print(line)
                            errors = True

                    lib_features[feature_name] = feature_name
                    if lib_features_and_level.get((feature_name, level)) is None:
                        # Add it to the observed features
                        lib_features_and_level[(feature_name, level)] = \
                            (since, path, line_num, line)
                    else:
                        # Verify that for this combination of feature_name and level the 'since'
                        # attribute matches.
                        (expected_since, source_path, source_line_num, source_line) = \
                            lib_features_and_level.get((feature_name, level))
                        if since != expected_since:
                            print("error: mismatch in %s feature '%s'" % (level, feature_name))
                            print("line %d of %s:" % (source_line_num, source_path))
                            print(source_line)
                            print("line %d of %s:" % (line_num, path))
                            print(line)
                            errors = True

                    # Verify that this lib feature doesn't duplicate a lang feature
                    if feature_name in language_feature_names:
                        print("error: lib feature '%s' duplicates a lang feature" % (feature_name))
                        print("line %d of %s:" % (line_num, path))
                        print(line)
                        errors = True

                else:
                    print("error: misformed stability attribute")
                    print("line %d of %s:" % (line_num, path))
                    print(line)
                    errors = True

# Merge data about both lists
# name, lang, lib, status, stable since

language_feature_stats = {}

for f in language_features:
    name = f[0]
    lang = True
    lib = False
    status = "unstable"
    stable_since = None

    if f[3] == "Accepted":
        status = "stable"
    if status == "stable":
        stable_since = f[1]

    language_feature_stats[name] = (name, lang, lib, status, stable_since)

lib_feature_stats = {}

for f in lib_features:
    name = f
    lang = False
    lib = True
    status = "unstable"
    stable_since = None

    is_stable = lib_features_and_level.get((name, "stable")) is not None
    is_unstable = lib_features_and_level.get((name, "unstable")) is not None

    if is_stable and is_unstable:
        print("error: feature '%s' is both stable and unstable" % (name))
        errors = True

    if is_stable:
        status = "stable"
        stable_since = lib_features_and_level[(name, "stable")][0]
    elif is_unstable:
        status = "unstable"

    lib_feature_stats[name] = (name, lang, lib, status, stable_since)

# Check for overlap in two sets
merged_stats = { }

for name in lib_feature_stats:
    if language_feature_stats.get(name) is not None:
        if not name in joint_features:
            print("error: feature '%s' is both a lang and lib feature but not whitelisted" % (name))
            errors = True
        lang_status = language_feature_stats[name][3]
        lib_status = lib_feature_stats[name][3]
        lang_stable_since = language_feature_stats[name][4]
        lib_stable_since = lib_feature_stats[name][4]

        if lang_status != lib_status and lib_status != "rustc_deprecated":
            print("error: feature '%s' has lang status %s " +
                  "but lib status %s" % (name, lang_status, lib_status))
            errors = True

        if lang_stable_since != lib_stable_since:
            print("error: feature '%s' has lang stable since %s " +
                  "but lib stable since %s" % (name, lang_stable_since, lib_stable_since))
            errors = True

        merged_stats[name] = (name, True, True, lang_status, lang_stable_since)

        del language_feature_stats[name]
        del lib_feature_stats[name]

if errors:
    sys.exit(1)

# Finally, display the stats
stats = {}
stats.update(language_feature_stats)
stats.update(lib_feature_stats)
stats.update(merged_stats)
lines = []
for s in stats:
    s = stats[s]
    type_ = "lang"
    if s[1] and s[2]:
        type_ = "lang/lib"
    elif s[2]:
        type_ = "lib"
    line = "{: <32}".format(s[0]) + \
           "{: <8}".format(type_) + \
           "{: <12}".format(s[3]) + \
           "{: <8}".format(str(s[4]))
    lines += [line]

lines.sort()

print
for line in lines:
    print("* " + line)
print
