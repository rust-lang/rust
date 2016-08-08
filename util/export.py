#!/usr/bin/env python
# Build the gh-pages

import re
import sys
import json

from lintlib import parse_all, log

lint_subheadline = re.compile(r'''^\*\*([\w\s]+?)[:?.!]?\*\*(.*)''')

CONF_TEMPLATE = """\
This lint has the following configuration variables:

* `%s: %s`: %s (defaults to `%s`)."""


def parse_lint_def(lint):
    lint_dict = {}
    lint_dict['id'] = lint.name
    lint_dict['level'] = lint.level
    lint_dict['docs'] = {}

    last_section = None

    for line in lint.doc:
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
            log.warn("Skipping comment line as it was not preceded by a heading")
            log.debug("in lint `%s`, line `%s`", lint.name, line)

        lint_dict['docs'][last_section] = \
            (lint_dict['docs'].get(last_section, "") + "\n" + text).strip()

    return lint_dict


def main():
    lintlist, configs = parse_all()
    lints = {}
    for lint in lintlist:
        lints[lint.name] = parse_lint_def(lint)
        if lint.name in configs:
            lints[lint.name]['docs']['Configuration'] = \
                CONF_TEMPLATE % configs[lint.name]

    outfile = sys.argv[1] if len(sys.argv) > 1 else "util/gh-pages/lints.json"
    with open(outfile, "w") as fp:
        json.dump(list(lints.values()), fp, indent=2)
        log.info("wrote JSON for great justice")


if __name__ == "__main__":
    main()
