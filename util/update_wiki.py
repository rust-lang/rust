#!/usr/bin/env python
# Generate the wiki Home.md page from the contained doc comments
# requires the checked out wiki in ../rust-clippy.wiki/
# with -c option, print a warning and set exit status 1 if the file would be
# changed.
import os
import re
import sys


level_re = re.compile(r'''(Forbid|Deny|Warn|Allow)''')


def parse_path(p="src"):
    d = {}
    for f in os.listdir(p):
        if f.endswith(".rs"):
            parse_file(d, os.path.join(p, f))
    return d

START = 0
LINT = 1


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
                else:
                    last_comment = []
            if not comment:
                l = line.strip()
                m = re.search(r"pub\s+([A-Z_]+)", l)

                if m:
                    name = m.group(1).lower()

                    while True:
                        m = re.search(level_re, line)
                        if m:
                            level = m.group(0)
                            break

                        line = next(rs)

                    print("found %s with level %s in %s" % (name, level, f))
                    d[name] = (level, last_comment)
                    last_comment = []
                    comment = True
                if "}" in l:
                    print("Warning: Missing Lint-Name in", f)
                    comment = True

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


template = """\n# `%s`

**Default level:** %s

%s"""


def write_wiki_page(d, f):
    keys = list(d.keys())
    keys.sort()
    with open(f, "w") as w:
        w.write(PREFIX)

        for level in ('Deny', 'Warn', 'Allow'):
            w.write("\n**Those lints are %s by default**:\n\n" % level)
            for k in keys:
                if d[k][0] == level:
                    w.write("[`%s`](#%s)\n" % (k, k))

        w.write(WARNING)
        for k in keys:
            w.write(template % (k, d[k][0], "".join(d[k][1])))


def check_wiki_page(d, f):
    errors = []
    with open(f) as w:
        for line in w:
            m = re.match("# `([a-z_]+)`", line)
            if m:
                v = d.pop(m.group(1), "()")
                if v == "()":
                    errors.append("Missing wiki entry: " + m.group(1))
    keys = list(d.keys())
    keys.sort()
    for k in keys:
        errors.append("Spurious wiki entry: " + k)
    if errors:
        print("\n".join(errors))
        sys.exit(1)


def main():
    d = parse_path()
    if "-c" in sys.argv:
        check_wiki_page(d, "../rust-clippy.wiki/Home.md")
    else:
        write_wiki_page(d, "../rust-clippy.wiki/Home.md")

if __name__ == "__main__":
    main()
