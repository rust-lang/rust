#!/usr/bin/env python
# Generate a Markdown table of all lints, and put it in README.md.
# With -n option, only print the new table to stdout.
# With -c option, print a warning and set exit status to 1 if a file would be
# changed.

import os
import re
import sys

declare_deprecated_lint_re = re.compile(r'''
    declare_deprecated_lint! \s* [{(] \s*
    pub \s+ (?P<name>[A-Z_][A-Z_0-9]*) \s*,\s*
    " (?P<desc>(?:[^"\\]+|\\.)*) " \s* [})]
''', re.VERBOSE | re.DOTALL)

declare_clippy_lint_re = re.compile(r'''
    declare_clippy_lint! \s* [{(] \s*
    pub \s+ (?P<name>[A-Z_][A-Z_0-9]*) \s*,\s*
    (?P<cat>[a-z_]+) \s*,\s*
    " (?P<desc>(?:[^"\\]+|\\.)*) " \s* [})]
''', re.VERBOSE | re.DOTALL)

nl_escape_re = re.compile(r'\\\n\s*')

docs_link = 'https://rust-lang-nursery.github.io/rust-clippy/master/index.html'


def collect(deprecated_lints, clippy_lints, fn):
    """Collect all lints from a file.

    Adds entries to the lints list as `(module, name, level, desc)`.
    """
    with open(fn) as fp:
        code = fp.read()

    for match in declare_deprecated_lint_re.finditer(code):
        # remove \-newline escapes from description string
        desc = nl_escape_re.sub('', match.group('desc'))
        deprecated_lints.append((os.path.splitext(os.path.basename(fn))[0],
                                match.group('name').lower(),
                                desc.replace('\\"', '"')))

    for match in declare_clippy_lint_re.finditer(code):
        # remove \-newline escapes from description string
        desc = nl_escape_re.sub('', match.group('desc'))
        cat = match.group('cat')
        clippy_lints[cat].append((os.path.splitext(os.path.basename(fn))[0],
                                  match.group('name').lower(),
                                  "allow",
                                  desc.replace('\\"', '"')))


def gen_group(lints):
    """Write lint group (list of all lints in the form module::NAME)."""
    for (module, name, _, _) in sorted(lints):
        yield '        %s::%s,\n' % (module, name.upper())


def gen_mods(lints):
    """Declare modules"""

    for module in sorted(set(lint[0] for lint in lints)):
        yield 'pub mod %s;\n' % module


def gen_deprecated(lints):
    """Declare deprecated lints"""

    for lint in lints:
        yield '    store.register_removed(\n'
        yield '        "%s",\n' % lint[1]
        yield '        "%s",\n' % lint[2]
        yield '    );\n'


def replace_region(fn, region_start, region_end, callback,
                   replace_start=True, write_back=True):
    """Replace a region in a file delimited by two lines matching regexes.

    A callback is called to write the new region.  If `replace_start` is true,
    the start delimiter line is replaced as well.  The end delimiter line is
    never replaced.
    """
    # read current content
    with open(fn) as fp:
        lines = list(fp)

    found = False

    # replace old region with new region
    new_lines = []
    in_old_region = False
    for line in lines:
        if in_old_region:
            if re.search(region_end, line):
                in_old_region = False
                new_lines.extend(callback())
                new_lines.append(line)
        elif re.search(region_start, line):
            if not replace_start:
                new_lines.append(line)
            # old region starts here
            in_old_region = True
            found = True
        else:
            new_lines.append(line)

    if not found:
        print "regex " + region_start + " not found"

    # write back to file
    if write_back:
        with open(fn, 'w') as fp:
            fp.writelines(new_lines)

    # if something changed, return true
    return lines != new_lines


def main(print_only=False, check=False):
    deprecated_lints = []
    clippy_lints = {
        "correctness": [],
        "style": [],
        "complexity": [],
        "perf": [],
        "restriction": [],
        "pedantic": [],
        "cargo": [],
        "nursery": [],
    }

    # check directory
    if not os.path.isfile('clippy_lints/src/lib.rs'):
        print('Error: call this script from clippy checkout directory!')
        return

    # collect all lints from source files
    for fn in os.listdir('clippy_lints/src'):
        if fn.endswith('.rs'):
            collect(deprecated_lints, clippy_lints,
                    os.path.join('clippy_lints', 'src', fn))

    # determine version
    with open('Cargo.toml') as fp:
        for line in fp:
            if line.startswith('version ='):
                clippy_version = line.split()[2].strip('"')
                break
        else:
            print('Error: version not found in Cargo.toml!')
            return

    all_lints = []
    clippy_lint_groups = [
        "correctness",
        "style",
        "complexity",
        "perf",
    ]
    clippy_lint_list = []
    for x in clippy_lint_groups:
        clippy_lint_list += clippy_lints[x]
    for _, value in clippy_lints.iteritems():
        all_lints += value

    if print_only:
        print_clippy_lint_groups = [
            "correctness",
            "style",
            "complexity",
            "perf",
            "pedantic",
            "nursery",
            "restriction"
        ]
        for group in print_clippy_lint_groups:
            sys.stdout.write('\n## ' + group + '\n')
            for (_, name, _, descr) in sorted(clippy_lints[x]):
                sys.stdout.write('* [' + name + '](https://rust-lang-nursery.github.io/rust-clippy/master/index.html#' + name + ') (' + descr + ')\n')
        return

    # update the lint counter in README.md
    changed = replace_region(
        'README.md',
        r'^\[There are \d+ lints included in this crate!\]\(https://rust-lang-nursery.github.io/rust-clippy/master/index.html\)$', "",
        lambda: ['[There are %d lints included in this crate!](https://rust-lang-nursery.github.io/rust-clippy/master/index.html)\n' %
                 (len(all_lints))],
        write_back=not check)

    # update the links in the CHANGELOG
    changed |= replace_region(
        'CHANGELOG.md',
        "<!-- begin autogenerated links to wiki -->",
        "<!-- end autogenerated links to wiki -->",
        lambda: ["[`{0}`]: {1}#{0}\n".format(l[1], docs_link) for l in
                 sorted(all_lints + deprecated_lints,
                        key=lambda l: l[1])],
        replace_start=False, write_back=not check)

    # update version of clippy_lints in Cargo.toml
    changed |= replace_region(
        'Cargo.toml', r'# begin automatic update', '# end automatic update',
        lambda: ['clippy_lints = { version = "%s", path = "clippy_lints" }\n' %
                 clippy_version],
        replace_start=False, write_back=not check)

    # update version of clippy_lints in Cargo.toml
    changed |= replace_region(
        'clippy_lints/Cargo.toml', r'# begin automatic update', '# end automatic update',
        lambda: ['version = "%s"\n' % clippy_version],
        replace_start=False, write_back=not check)

    # update the `pub mod` list
    changed |= replace_region(
        'clippy_lints/src/lib.rs', r'begin lints modules', r'end lints modules',
        lambda: gen_mods(all_lints),
        replace_start=False, write_back=not check)

    # same for "clippy_*" lint collections
    changed |= replace_region(
        'clippy_lints/src/lib.rs', r'reg.register_lint_group\("clippy"', r'\]\);',
        lambda: gen_group(clippy_lint_list),
        replace_start=False, write_back=not check)

    for key, value in clippy_lints.iteritems():
        # same for "clippy_*" lint collections
        changed |= replace_region(
            'clippy_lints/src/lib.rs', r'reg.register_lint_group\("clippy_' + key + r'"', r'\]\);',
            lambda: gen_group(value),
            replace_start=False, write_back=not check)

    # same for "deprecated" lint collection
    changed |= replace_region(
        'clippy_lints/src/lib.rs', r'let mut store', r'end deprecated lints',
        lambda: gen_deprecated(deprecated_lints),
        replace_start=False,
        write_back=not check)

    if check and changed:
        print('Please run util/update_lints.py to regenerate lints lists.')
        return 1


if __name__ == '__main__':
    sys.exit(main(print_only='-n' in sys.argv, check='-c' in sys.argv))
