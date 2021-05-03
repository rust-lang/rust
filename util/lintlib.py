# Common utils for the several housekeeping scripts.

import os
import re
import collections

import logging as log
log.basicConfig(level=log.INFO, format='%(levelname)s: %(message)s')

Lint = collections.namedtuple('Lint', 'name level doc sourcefile group')
Config = collections.namedtuple('Config', 'name ty doc default')

lintname_re = re.compile(r'''pub\s+([A-Z_][A-Z_0-9]*)''')
group_re = re.compile(r'''\s*([a-z_][a-z_0-9]+)''')
conf_re = re.compile(r'''define_Conf! {\n([^}]*)\n}''', re.MULTILINE)
confvar_re = re.compile(
    r'''/// Lint: ([\w,\s]+)\. (.*)\n\s*\(([^:]+):\s*([^\s=]+)\s*=\s*([^\.\)]+).*\),''', re.MULTILINE)
comment_re = re.compile(r'''\s*/// ?(.*)''')

lint_levels = {
    "correctness": 'Deny',
    "style": 'Warn',
    "complexity": 'Warn',
    "perf": 'Warn',
    "restriction": 'Allow',
    "pedantic": 'Allow',
    "nursery": 'Allow',
    "cargo": 'Allow',
}


def parse_lints(lints, filepath):
    comment = []
    clippy = False
    deprecated = False
    name = ""

    with open(filepath) as fp:
        for line in fp:
            if clippy or deprecated:
                m = lintname_re.search(line)
                if m:
                    name = m.group(1).lower()
                    line = next(fp)

                    if deprecated:
                        level = "Deprecated"
                        group = "deprecated"
                    else:
                        while True:
                            g = group_re.search(line)
                            if g:
                                group = g.group(1).lower()
                                level = lint_levels.get(group, None)
                                break
                            line = next(fp)

                    if level is None:
                        continue

                    log.info("found %s with level %s in %s",
                             name, level, filepath)
                    lints.append(Lint(name, level, comment, filepath, group))
                    comment = []

                    clippy = False
                    deprecated = False
                    name = ""
                else:
                    m = comment_re.search(line)
                    if m:
                        comment.append(m.group(1))
            elif line.startswith("declare_clippy_lint!"):
                clippy = True
                deprecated = False
            elif line.startswith("declare_deprecated_lint!"):
                clippy = False
                deprecated = True
            elif line.startswith("declare_lint!"):
                import sys
                print(
                    "don't use `declare_lint!` in Clippy, "
                    "use `declare_clippy_lint!` instead"
                )
                sys.exit(42)


def parse_configs(path):
    configs = {}
    with open(os.path.join(path, 'utils/conf.rs')) as fp:
        contents = fp.read()

    match = re.search(conf_re, contents)
    confvars = re.findall(confvar_re, match.group(1))

    for (lints, doc, name, ty, default) in confvars:
        for lint in lints.split(','):
            configs[lint.strip().lower()] = Config(name.replace("_", "-"), ty, doc, default)
    return configs


def parse_all(path="clippy_lints/src"):
    lints = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            if fn.endswith('.rs'):
                parse_lints(lints, os.path.join(root, fn))

    log.info("got %s lints", len(lints))

    configs = parse_configs(path)
    log.info("got %d configs", len(configs))

    return lints, configs
