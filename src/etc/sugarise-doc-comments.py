#!/usr/bin/env python
#
# Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

#
# this script attempts to turn doc comment attributes (#[doc = "..."])
# into sugared-doc-comments (/** ... */ and /// ...)
#
# it sugarises all .rs/.rc files underneath the working directory
#

import sys, os, fnmatch, re


DOC_PATTERN = '^(?P<indent>[\\t ]*)#\\[(\\s*)doc(\\s*)=' + \
              '(\\s*)"(?P<text>(\\"|[^"])*?)"(\\s*)\\]' + \
              '(?P<semi>;)?'

ESCAPES = [("\\'", "'"),
           ('\\"', '"'),
           ("\\n", "\n"),
           ("\\r", "\r"),
           ("\\t", "\t")]


def unescape(s):
    for (find, repl) in ESCAPES:
        s = s.replace(find, repl)
    return s


def block_trim(s):
    lns = s.splitlines()

    # remove leading/trailing whitespace-lines
    while lns and not lns[0].strip():
        lns = lns[1:]
    while lns and not lns[-1].strip():
        lns = lns[:-1]

    # remove leading horizontal whitespace
    n = sys.maxint
    for ln in lns:
        if ln.strip():
            n = min(n, len(re.search('^\s*', ln).group()))
    if n != sys.maxint:
        lns = [ln[n:] for ln in lns]

    # strip trailing whitespace
    lns = [ln.rstrip() for ln in lns]

    return lns


def replace_doc(m):
    indent = m.group('indent')
    text = block_trim(unescape(m.group('text')))

    if len(text) > 1:
        inner = '!' if m.group('semi') else '*'
        starify = lambda s: indent + ' *' + (' ' + s if s else '')
        text = '\n'.join(map(starify, text))
        repl = indent + '/*' + inner + '\n' + text + '\n' + indent + ' */'
    else:
        inner = '!' if m.group('semi') else '/'
        repl = indent + '//' + inner + ' ' + text[0]

    return repl


def sugarise_file(path):
    s = open(path).read()

    r = re.compile(DOC_PATTERN, re.MULTILINE | re.DOTALL)
    ns = re.sub(r, replace_doc, s)

    if s != ns:
        open(path, 'w').write(ns)


for (dirpath, dirnames, filenames) in os.walk('.'):
    for name in fnmatch.filter(filenames, '*.r[sc]'):
        sugarise_file(os.path.join(dirpath, name))
