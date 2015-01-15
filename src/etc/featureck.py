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

import sys, os, re

src_dir = sys.argv[1]

# Features that are allowed to exist in both the language and the library
joint_features = [ "on_unimpleented" ]

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
            line = line.replace("(", "").replace("),", "").replace(")", "")
            parts = line.split(",")
            if len(parts) != 3:
                print "unexpected number of components in line: " + original_line
                sys.exit(1)
            feature_name = parts[0].strip().replace('"', "")
            since = parts[1].strip().replace('"', "")
            status = parts[2].strip()
            assert len(feature_name) > 0
            assert len(since) > 0
            assert len(status) > 0

            language_feature_names += [feature_name]
            language_features += [(feature_name, since, status)]

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
        with open(path, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                level = None
                if "[unstable(" in line:
                    level = "unstable"
                elif "[stable(" in line:
                    level = "stable"
                elif "[deprecated(" in line:
                    level = "deprecated"
                else:
                    continue

                # This is a stability attribute. For the purposes of this
                # script we expect both the 'feature' and 'since' attributes on
                # the same line, e.g.
                # `#[unstable(feature = "foo", since = "1.0.0")]`

                p = re.compile('feature *= *"(\w*)".*since *= *"([\w\.]*)"')
                m = p.search(line)
                if not m is None:
                    feature_name = m.group(1)
                    since = m.group(2)
                    lib_features[feature_name] = feature_name
                    if lib_features_and_level.get((feature_name, level)) is None:
                        # Add it to the observed features
                        lib_features_and_level[(feature_name, level)] = (since, path, line_num, line)
                    else:
                        # Verify that for this combination of feature_name and level the 'since'
                        # attribute matches.
                        (expected_since, source_path, source_line_num, source_line) = \
                            lib_features_and_level.get((feature_name, level))
                        if since != expected_since:
                            print "mismatch in " + level + " feature '" + feature_name + "'"
                            print "line " + str(source_line_num) + " of " + source_path + ":"
                            print source_line
                            print "line " + str(line_num) + " of " + path + ":"
                            print line
                            errors = True

                    # Verify that this lib feature doesn't duplicate a lang feature
                    if feature_name in language_feature_names:
                        print "lib feature '" + feature_name + "' duplicates a lang feature"
                        print "line " + str(line_num) + " of " + path + ":"
                        print line
                        errors = True

                else:
                    print "misformed stability attribute"
                    print "line " + str(line_num) + " of " + path + ":"
                    print line
                    errors = True

# Merge data about both lists
# name, lang, lib, status, stable since, partially deprecated

language_feature_stats = {}

for f in language_features:
    name = f[0]
    lang = True
    lib = False
    status = "unstable"
    stable_since = None
    partially_deprecated = False
    
    if f[2] == "Accepted":
        status = "stable"
    if status == "stable":
        stable_since = f[1]

    language_feature_stats[name] = (name, lang, lib, status, stable_since, \
                                    partially_deprecated)

lib_feature_stats = {}

for f in lib_features:
    name = f
    lang = False
    lib = True
    status = "unstable"
    stable_since = None
    partially_deprecated = False

    is_stable = lib_features_and_level.get((name, "stable")) is not None
    is_unstable = lib_features_and_level.get((name, "unstable")) is not None
    is_deprecated = lib_features_and_level.get((name, "deprecated")) is not None

    if is_stable and is_unstable:
        print "feature '" + name + "' is both stable and unstable"
        errors = True

    if is_stable:
        status = "stable"
        stable_since = lib_features_and_level[(name, "stable")][0]
    elif is_unstable:
        status = "unstable"
        stable_since = lib_features_and_level[(name, "unstable")][0]
    elif is_deprecated:
        status = "deprecated"

    if (is_stable or is_unstable) and is_deprecated:
        partially_deprecated = True

    lib_feature_stats[name] = (name, lang, lib, status, stable_since, \
                               partially_deprecated)

# Check for overlap in two sets
merged_stats = { }

for name in lib_feature_stats:
    if language_feature_stats.get(name) is not None:
        if not name in joint_features:
            print "feature '" + name + "' is both a lang and lib feature but not whitelisted"
            errors = True
        lang_status = lang_feature_stats[name][3]
        lib_status = lib_feature_stats[name][3]
        lang_stable_since = lang_feature_stats[name][4]
        lib_stable_since = lib_feature_stats[name][4]
        lang_partially_deprecated = lang_feature_stats[name][5]
        lib_partially_deprecated = lib_feature_stats[name][5]

        if lang_status != lib_status and lib_status != "deprecated":
            print "feature '" + name + "' has lang status " + lang_status + \
                  " but lib status " + lib_status
            errors = True

        partially_deprecated = lang_partially_deprecated or lib_partially_deprecated
        if lib_status == "deprecated" and lang_status != "deprecated":
            partially_deprecated = True

        if lang_stable_since != lib_stable_since:
            print "feature '" + name + "' has lang stable since " + lang_stable_since + \
                  " but lib stable since " + lib_stable_since
            errors = True

        merged_stats[name] = (name, True, True, lang_status, lang_stable_since, \
                              partially_deprecated)

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
    line = s[0] + ",\t\t\t" + type_ + ",\t" + s[3] + ",\t" + str(s[4])
    line = "{: <32}".format(s[0]) + \
           "{: <8}".format(type_) + \
           "{: <12}".format(s[3]) + \
           "{: <8}".format(str(s[4]))
    if s[5]:
        line += "(partially deprecated)"
    lines += [line]

lines.sort()

print
print "Rust feature summary:"
print
for line in lines:
    print line
print

