#!/usr/bin/env python
# Generate the wiki Home.md page from the contained doc comments
# requires the checked out wiki in ../rust-clippy.wiki/
# with -c option, print a warning and set exit status 1 if the file would be
# changed.
import os
import re
import sys


level_re = re.compile(r'''(Forbid|Deny|Warn|Allow)''')
conf_re = re.compile(r'''define_Conf! {\n([^}]*)\n}''', re.MULTILINE)
confvar_re = re.compile(r'''/// Lint: (\w+). (.*).*\n *\("([^"]*)", (?:[^,]*), (.*) => (.*)\),''')


def parse_path(p="clippy_lints/src"):
    d = {}
    for f in os.listdir(p):
        if f.endswith(".rs"):
            parse_file(d, os.path.join(p, f))
    return (d, parse_conf(p))


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
                    while not deprecated:
                        m = re.search(level_re, line)
                        if m:
                            level = m.group(0)
                            break

                        line = next(rs)

                    if deprecated:
                        level = "Deprecated"

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

conf_template = """
**Configuration:** This lint has the following configuration variables:

* `%s: %s`: %s (defaults to `%s`).
"""


def level_message(level):
    if level == "Deprecated":
        return "\n**Those lints are deprecated**:\n\n"
    else:
        return "\n**Those lints are %s by default**:\n\n" % level


def write_wiki_page(d, c, f):
    keys = list(d.keys())
    keys.sort()
    with open(f, "w") as w:
        w.write(PREFIX)

        for level in ('Deny', 'Warn', 'Allow', 'Deprecated'):
            w.write(level_message(level))
            for k in keys:
                if d[k][0] == level:
                    w.write("[`%s`](#%s)\n" % (k, k))

        w.write(WARNING)
        for k in keys:
            w.write(template % (k, d[k][0], "".join(d[k][1])))

            if k in c:
                w.write(conf_template % c[k])


def check_wiki_page(d, c, f):
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
    (d, c) = parse_path()
    if "-c" in sys.argv:
        check_wiki_page(d, c, "../rust-clippy.wiki/Home.md")
    else:
        write_wiki_page(d, c, "../rust-clippy.wiki/Home.md")

if __name__ == "__main__":
    main()
