#!/usr/bin/env python2

# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from __future__ import absolute_import, division, print_function
import argparse
from collections import defaultdict
import csv
import datetime
import urllib2

BASE_URL = 'http://www.unicode.org/Public/6.3.0/ucd/'
DATA = 'UnicodeData.txt'
SCRIPTS = 'Scripts.txt'

# Mapping taken from Table 12 from:
# http://www.unicode.org/reports/tr44/#General_Category_Values
expanded_categories = {
    'Lu': ['LC', 'L'], 'Ll': ['LC', 'L'], 'Lt': ['LC', 'L'],
    'Lm': ['L'], 'Lo': ['L'],
    'Mn': ['M'], 'Mc': ['M'], 'Me': ['M'],
    'Nd': ['N'], 'Nl': ['N'], 'No': ['No'],
    'Pc': ['P'], 'Pd': ['P'], 'Ps': ['P'], 'Pe': ['P'],
    'Pi': ['P'], 'Pf': ['P'], 'Po': ['P'],
    'Sm': ['S'], 'Sc': ['S'], 'Sk': ['S'], 'So': ['S'],
    'Zs': ['Z'], 'Zl': ['Z'], 'Zp': ['Z'],
    'Cc': ['C'], 'Cf': ['C'], 'Cs': ['C'], 'Co': ['C'], 'Cn': ['C'],
}


def as_4byte_uni(n):
    s = hex(n)[2:]
    return '\\U%s%s' % ('0' * (8 - len(s)), s)


def expand_cat(c):
    return expanded_categories.get(c, []) + [c]


def is_valid_unicode(n):
    return 0 <= n <= 0xD7FF or 0xE000 <= n <= 0x10FFFF


def read_cats(f):
    assigned = defaultdict(list)
    for row in csv.reader(f, delimiter=';'):
        (hex, cats) = (int(row[0], 16), expand_cat(row[2]))
        if not is_valid_unicode(hex):
            continue
        for cat in cats:
            assigned[cat].append(hex)
    return assigned


def read_scripts(f):
    assigned = defaultdict(list)
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        hexes, name = map(str.strip, line.split(';'))[:2]
        name = name[:name.index('#')].strip()
        if '..' not in hexes:
            hex = int(hexes, 16)
            if is_valid_unicode(hex):
                assigned[name].append(hex)
        else:
            hex1, hex2 = map(lambda s: int(s, 16), hexes.split('..'))
            for hex in xrange(hex1, hex2 + 1):
                if is_valid_unicode(hex):
                    assigned[name].append(hex)
    return assigned


def group(letters):
    letters = sorted(set(letters))
    grouped = []
    cur_start = letters.pop(0)
    cur_end = cur_start
    for letter in letters:
        assert letter > cur_end, \
            'cur_end: %s, letter: %s' % (hex(cur_end), hex(letter))

        if letter == cur_end + 1:
            cur_end = letter
        else:
            grouped.append((cur_start, cur_end))
            cur_start, cur_end = letter, letter
    grouped.append((cur_start, cur_end))
    return grouped


def ranges_to_rust(rs):
    rs = ("('%s', '%s')" % (as_4byte_uni(s), as_4byte_uni(e)) for s, e in rs)
    return ',\n    '.join(rs)


def groups_to_rust(groups):
    rust_groups = []
    for group_name in sorted(groups):
        rust_groups.append('("%s", &[\n    %s\n    ]),'
                           % (group_name, ranges_to_rust(groups[group_name])))
    return '\n'.join(rust_groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Unicode character class tables.')
    aa = parser.add_argument
    aa('--local', action='store_true',
       help='When set, Scripts.txt and UnicodeData.txt will be read from '
            'the CWD.')
    aa('--base-url', type=str, default=BASE_URL,
       help='The base URL to use for downloading Unicode data files.')
    args = parser.parse_args()

    if args.local:
        cats = read_cats(open(DATA))
        scripts = read_scripts(open(SCRIPTS))
    else:
        cats = read_cats(urllib2.urlopen(args.base_url + '/' + DATA))
        scripts = read_scripts(urllib2.urlopen(args.base_url + '/' + SCRIPTS))

    # Get Rust code for all Unicode general categories and scripts.
    combined = dict(cats, **scripts)
    unigroups = groups_to_rust({k: group(letters)
                                for k, letters in combined.items()})

    # Now get Perl character classes that are Unicode friendly.
    perld = range(ord('0'), ord('9') + 1)
    dgroups = ranges_to_rust(group(perld + cats['Nd'][:]))

    perls = map(ord, ['\t', '\n', '\x0C', '\r', ' '])
    sgroups = ranges_to_rust(group(perls + cats['Z'][:]))

    low, up = (range(ord('a'), ord('z') + 1), range(ord('A'), ord('Z') + 1))
    perlw = [ord('_')] + perld + low + up
    wgroups = ranges_to_rust(group(perlw + cats['L'][:]))

    tpl = '''// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// DO NOT EDIT. Automatically generated by 'src/etc/regexp-unicode-tables'
// on {date}.

use parse::{{Class, NamedClasses}};

pub static UNICODE_CLASSES: NamedClasses = &[

{groups}

];

pub static PERLD: Class = &[
    {dgroups}
];

pub static PERLS: Class = &[
    {sgroups}
];

pub static PERLW: Class = &[
    {wgroups}
];
'''
    now = datetime.datetime.now()
    print(tpl.format(date=str(now), groups=unigroups,
                     dgroups=dgroups, sgroups=sgroups, wgroups=wgroups))
