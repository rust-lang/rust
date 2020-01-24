#!/usr/bin/env python

# Build the gh-pages

from collections import OrderedDict
import re
import sys
import json

from lintlib import parse_all, log

lint_subheadline = re.compile(r'''^\*\*([\w\s]+?)[:?.!]?\*\*(.*)''')
rust_code_block = re.compile(r'''```rust.+?```''', flags=re.DOTALL)

CONF_TEMPLATE = """\
This lint has the following configuration variables:

* `%s: %s`: %s (defaults to `%s`)."""


def parse_code_block(match):
    lines = []

    for line in match.group(0).split('\n'):
        if not line.startswith('# '):
            lines.append(line)

    return '\n'.join(lines)


def parse_lint_def(lint):
    lint_dict = {}
    lint_dict['id'] = lint.name
    lint_dict['group'] = lint.group
    lint_dict['level'] = lint.level
    lint_dict['docs'] = OrderedDict()

    last_section = None

    for line in lint.doc:
        match = re.match(lint_subheadline, line)
        if match:
            last_section = match.groups()[0]
            text = match.groups()[1]
        else:
            text = line

        if not last_section:
            log.warning("Skipping comment line as it was not preceded by a heading")
            log.debug("in lint `%s`, line `%s`", lint.name, line)

        if last_section not in lint_dict['docs']:
            lint_dict['docs'][last_section] = ""

        lint_dict['docs'][last_section] += text + "\n"

    for section in lint_dict['docs']:
        lint_dict['docs'][section] = re.sub(rust_code_block, parse_code_block, lint_dict['docs'][section].strip())

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
        lints = list(lints.values())
        lints.sort(key=lambda x: x['id'])
        json.dump(lints, fp, indent=2)
        log.info("wrote JSON for great justice")


if __name__ == "__main__":
    main()
