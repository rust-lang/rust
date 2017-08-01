# Common utils for the several housekeeping scripts.

import os
import re
import collections

import logging as log
log.basicConfig(level=log.INFO, format='%(levelname)s: %(message)s')

Lint = collections.namedtuple('Lint', 'name level doc sourcefile')
Config = collections.namedtuple('Config', 'name ty doc default')

lintname_re = re.compile(r'''pub\s+([A-Z_][A-Z_0-9]*)''')
level_re = re.compile(r'''(Forbid|Deny|Warn|Allow)''')
conf_re = re.compile(r'''define_Conf! {\n([^}]*)\n}''', re.MULTILINE)
confvar_re = re.compile(
    r'''/// Lint: (\w+). (.*).*\n\s*\([^,]+,\s+"([^"]+)",\s+([^=\)]+)=>\s+(.*)\),''', re.MULTILINE)


def parse_lints(lints, filepath):
    last_comment = []
    comment = True

    with open(filepath) as fp:
        for line in fp:
            if comment:
                if line.startswith("/// "):
                    last_comment.append(line[4:])
                elif line.startswith("///"):
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
                m = lintname_re.search(line)
                if m:
                    name = m.group(1).lower()

                    if deprecated:
                        level = "Deprecated"
                    elif restriction:
                        level = "Allow"
                    else:
                        while True:
                            m = level_re.search(line)
                            if m:
                                level = m.group(0)
                                break
                            line = next(fp)

                    log.info("found %s with level %s in %s",
                             name, level, filepath)
                    lints.append(Lint(name, level, last_comment, filepath))
                    last_comment = []
                    comment = True
                if "}" in line:
                    log.warn("Warning: missing Lint-Name in %s", filepath)
                    comment = True


def parse_configs(path):
    configs = {}
    with open(os.path.join(path, 'utils/conf.rs')) as fp:
        contents = fp.read()

    match = re.search(conf_re, contents)
    confvars = re.findall(confvar_re, match.group(1))

    for (lint, doc, name, default, ty) in confvars:
        configs[lint.lower()] = Config(name, ty, doc, default)

    return configs


def parse_all(path="clippy_lints/src"):
    lints = []
    for filename in os.listdir(path):
        if filename.endswith(".rs"):
            parse_lints(lints, os.path.join(path, filename))
    log.info("got %s lints", len(lints))

    configs = parse_configs(path)
    log.info("got %d configs", len(configs))

    return lints, configs
