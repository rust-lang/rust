#!/usr/bin/env python
# Generate the wiki Home.md page from the contained doc comments
# requires the checked out wiki in ../rust-clippy.wiki/
# with -c option, print a warning and set exit status 1 if the file would be
# changed.

import re
import sys

from lintlib import log, parse_all

PREFIX = """Welcome to the rust-clippy wiki!

Here we aim to collect further explanations on the lints clippy provides. So \
without further ado:
"""

WARNING = """
# A word of warning

Clippy works as a *plugin* to the compiler, which means using an unstable \
internal API. We have gotten quite good at keeping pace with the API \
evolution, but the consequence is that clippy absolutely needs to be compiled \
with the version of `rustc` it will run on, otherwise you will get strange \
errors of missing symbols.

"""

TEMPLATE = """\n# `%s`

**Default level:** %s

%s"""

CONF_TEMPLATE = """
**Configuration:** This lint has the following configuration variables:

* `%s: %s`: %s (defaults to `%s`).
"""


def level_message(level):
    if level == "Deprecated":
        return "\n**Those lints are deprecated**:\n\n"
    else:
        return "\n**Those lints are %s by default**:\n\n" % level


def write_wiki_page(lints, configs, filepath):
    lints.sort()
    with open(filepath, "w") as fp:
        fp.write(PREFIX)

        for level in ('Deny', 'Warn', 'Allow', 'Deprecated'):
            fp.write(level_message(level))
            for lint in lints:
                if lint.level == level:
                    fp.write("[`%s`](#%s)\n" % (lint.name, lint.name))

        fp.write(WARNING)
        for lint in lints:
            fp.write(TEMPLATE % (lint.name, lint.level, "".join(lint.doc)))

            if lint.name in configs:
                fp.write(CONF_TEMPLATE % configs[lint.name])


def check_wiki_page(lints, configs, filepath):
    lintdict = dict((lint.name, lint) for lint in lints)
    errors = False
    with open(filepath) as fp:
        for line in fp:
            m = re.match("# `([a-z_0-9]+)`", line)
            if m:
                v = lintdict.pop(m.group(1), None)
                if v is None:
                    log.error("Spurious wiki entry: %s", m.group(1))
                    errors = True
    for n in sorted(lintdict):
        log.error("Missing wiki entry: %s", n)
        errors = True
    if errors:
        return 1


def main():
    lints, configs = parse_all()
    if "-c" in sys.argv:
        check_wiki_page(lints, configs, "../rust-clippy.wiki/Home.md")
    else:
        write_wiki_page(lints, configs, "../rust-clippy.wiki/Home.md")


if __name__ == "__main__":
    main()
