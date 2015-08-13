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


def collect(lints, fp):
    code = fp.read()
    for match in declare_lint_re.finditer(code):
        # remove \-newline escapes from description string
        desc = nl_escape_re.sub('', match.group('desc'))
        lints.append((match.group('name').lower(),
                      match.group('level').lower(),
                      desc.replace('\\"', '"')))


def write_tbl(lints, fp):
    # first and third column widths
    w_name = max(len(l[0]) for l in lints)
    w_desc = max(len(l[2]) for l in lints)
    # header and underline
    fp.write('%-*s | default | meaning\n' % (w_name, 'name'))
    fp.write('%s-|-%s-|-%s\n' % ('-' * w_name, '-' * 7, '-' * w_desc))
    # one table row per lint
    for (name, default, meaning) in sorted(lints):
        fp.write('%-*s | %-7s | %s\n' % (w_name, name, default, meaning))


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
                with open(os.path.join(root, fn)) as fp:
                    collect(lints, fp)

    if print_only:
        write_tbl(lints, sys.stdout)
        return

    # read current README.md content
    with open('README.md') as fp:
        lines = list(fp)

    # replace old table with new table
    with open('README.md', 'w') as fp:
        in_old_tbl = False
        for line in lines:
            if line.replace(' ', '').strip() == 'name|default|meaning':
                # old table starts here
                write_tbl(lints, fp)
                in_old_tbl = True
            if in_old_tbl:
                # the old table is finished by an empty line
                if line.strip():
                    continue
                in_old_tbl = False
            fp.write(line)


if __name__ == '__main__':
    main(print_only='-n' in sys.argv)
