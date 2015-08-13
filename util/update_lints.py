# Generate a Markdown table of all lints, and put it in README.md.
# With -n option, only print the new table to stdout.

import os
import re
import sys

declare_lint_re = re.compile(r'''
    declare_lint! \s* [{(] \s*
    pub \s+ (?P<name>[A-Z_]+) \s*,\s*
    (?P<level>Forbid|Deny|Warn|Allow) \s*,\s*
    " (?P<desc>(?:[^"\\]+|\\.)*) " \s* [})]
''', re.X | re.S)

nl_escape_re = re.compile(r'\\\n\s*')


def collect(lints, fn):
    """Collect all lints from a file.

    Adds entries to the lints list as `(module, name, level, desc)`.
    """
    with open(fn) as fp:
        code = fp.read()
    for match in declare_lint_re.finditer(code):
        # remove \-newline escapes from description string
        desc = nl_escape_re.sub('', match.group('desc'))
        lints.append((os.path.splitext(os.path.basename(fn))[0],
                      match.group('name').lower(),
                      match.group('level').lower(),
                      desc.replace('\\"', '"')))


def write_tbl(lints, fp):
    """Write lint table in Markdown format."""
    # first and third column widths
    w_name = max(len(l[1]) for l in lints)
    w_desc = max(len(l[3]) for l in lints)
    # header and underline
    fp.write('%-*s | default | meaning\n' % (w_name, 'name'))
    fp.write('%s-|-%s-|-%s\n' % ('-' * w_name, '-' * 7, '-' * w_desc))
    # one table row per lint
    for (_, name, default, meaning) in sorted(lints, key=lambda l: l[1]):
        fp.write('%-*s | %-7s | %s\n' % (w_name, name, default, meaning))


def write_group(lints, fp):
    """Write lint group (list of all lints in the form module::NAME)."""
    for (module, name, _, _) in sorted(lints):
        fp.write('        %s::%s,\n' % (module, name.upper()))


def replace_region(fn, region_start, region_end, callback,
                   replace_start=True):
    """Replace a region in a file delimited by two lines matching regexes.

    A callback is called to write the new region.  If `replace_start` is true,
    the start delimiter line is replaced as well.  The end delimiter line is
    never replaced.
    """
    # read current content
    with open(fn) as fp:
        lines = list(fp)

    # replace old region with new region
    with open(fn, 'w') as fp:
        in_old_region = False
        for line in lines:
            if in_old_region:
                if re.search(region_end, line):
                    in_old_region = False
                    fp.write(line)
            elif re.search(region_start, line):
                if not replace_start:
                    fp.write(line)
                # old region starts here
                in_old_region = True
                callback(fp)
            else:
                fp.write(line)


def main(print_only=False):
    lints = []

    # check directory
    if not os.path.isfile('src/lib.rs'):
        print('Error: call this script from clippy checkout directory!')
        return

    # collect all lints from source files
    for root, dirs, files in os.walk('src'):
        for fn in files:
            if fn.endswith('.rs'):
                collect(lints, os.path.join(root, fn))

    if print_only:
        write_tbl(lints, sys.stdout)
        return

    # replace table in README.md
    replace_region('README.md', r'^name +\|', '^$', lambda fp: write_tbl(lints, fp))

    # same for "clippy" lint collection
    replace_region('src/lib.rs', r'reg.register_lint_group\("clippy"', r'\]\);',
                   lambda fp: write_group(lints, fp), replace_start=False)


if __name__ == '__main__':
    main(print_only='-n' in sys.argv)
