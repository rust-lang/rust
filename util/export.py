#!/usr/bin/env python
# Build the gh-pages

import json
import os
import re
import sys


level_re = re.compile(r'''(Forbid|Deny|Warn|Allow)''')
conf_re = re.compile(r'''define_Conf! {\n([^}]*)\n}''', re.MULTILINE)
confvar_re = re.compile(r'''/// Lint: (\w+). (.*).*\n *\("([^"]*)", (?:[^,]*), (.*) => (.*)\),''')
lint_subheadline = re.compile(r'''^\*\*([\w\s]+?)[:?.!]?\*\*(.*)''')

conf_template = """
This lint has the following configuration variables:

* `%s: %s`: %s (defaults to `%s`).
"""


# TODO: actual logging
def warn(*args):
    print(*args)


def debug(*args):
    print(*args)


def info(*args):
    print(*args)


def parse_path(p="clippy_lints/src"):
    lints = []
    for f in os.listdir(p):
        if f.endswith(".rs"):
            parse_file(lints, os.path.join(p, f))

    conf = parse_conf(p)
    info(conf)

    for lint_id in conf:
        lint = next(l for l in lints if l['id'] == lint_id)
        if lint:
            lint['docs']['Configuration'] = (conf_template % conf[lint_id]).strip()

    return lints


def parse_conf(p):
    c = {}
    with open(p + '/utils/conf.rs') as f:
        f = f.read()

        m = re.search(conf_re, f)
        m = m.groups()[0]

        m = re.findall(confvar_re, m)

        for (lint, doc, name, default, ty) in m:
            c[lint.lower()] = (name, ty, doc, default)

    return c


def parseLintDef(level, comment, name):
    lint = {}
    lint['id'] = name
    lint['level'] = level
    lint['docs'] = {}

    last_section = None

    for line in comment:
        if len(line.strip()) == 0:
            continue

        match = re.match(lint_subheadline, line)
        if match:
            last_section = match.groups()[0]
        if match:
            text = match.groups()[1]
        else:
            text = line

        if not last_section:
            warn("Skipping comment line as it was not preceded by a heading")
            debug("in lint `%s`, line `%s`" % name, line)

        lint['docs'][last_section] = (lint['docs'].get(last_section, "") + "\n" + text).strip()

    return lint


def parse_file(d, f):
    last_comment = []
    comment = True

    with open(f) as rs:
        for line in rs:
            if comment:
                if line.startswith("///"):
                    if line.startswith("/// "):
                        last_comment.append(line[4:])
                    else:
                        last_comment.append(line[3:])
                elif line.startswith("declare_lint!"):
                    comment = False
                    deprecated = False
                    restriction = False
                elif line.startswith("declare_restriction_lint!"):
                    comment = False
                    deprecated = False
                    restriction = True
                elif line.startswith("declare_deprecated_lint!"):
                    comment = False
                    deprecated = True
                else:
                    last_comment = []
            if not comment:
                l = line.strip()
                m = re.search(r"pub\s+([A-Z_][A-Z_0-9]*)", l)

                if m:
                    name = m.group(1).lower()

                    # Intentionally either a never looping or infinite loop
                    while not deprecated and not restriction:
                        m = re.search(level_re, line)
                        if m:
                            level = m.group(0)
                            break

                        line = next(rs)

                    if deprecated:
                        level = "Deprecated"
                    elif restriction:
                        level = "Allow"

                    info("found %s with level %s in %s" % (name, level, f))
                    d.append(parseLintDef(level, last_comment, name=name))
                    last_comment = []
                    comment = True
                if "}" in l:
                    warn("Warning: Missing Lint-Name in", f)
                    comment = True


def main():
    lints = parse_path()
    info("got %s lints" % len(lints))

    outdir = sys.argv[1] if len(sys.argv) > 1 else "util/gh-pages/lints.json"
    with open(outdir, "w") as file:
        json.dump(lints, file, indent=2)
        info("wrote JSON for great justice")

if __name__ == "__main__":
    main()
