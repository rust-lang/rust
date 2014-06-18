#!/bin/env python
#
# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import sys
import subprocess
import re


def main():
    if len(sys.argv) <= 1:
        print('Usage: %s [ --apply ] filename1.rs filename2.rs ...'
              % sys.argv[0])
    elif sys.argv[1] == '--apply':
        for filename in sys.argv[2:]:
            patch(filename)
    else:
        for filename in sys.argv[1:]:
            diff(filename)


def patch(filename):
    source = read(filename)
    rewritten = rewrite_bytes_macros(source)
    if rewritten is not None and rewritten != source:
        write(filename, rewritten)


def diff(filename):
    rewritten = rewrite_bytes_macros(read(filename))
    if rewritten is not None:
        p = subprocess.Popen(['diff', '-u', filename, '-'],
                             stdin=subprocess.PIPE)
        p.stdin.write(rewritten)
        p.stdin.close()
        p.wait()


def read(filename):
    with open(filename, 'rb') as f:
        return f.read()


def write(filename, content):
    with open(filename, 'wb') as f:
        f.write(content)


def rewrite_bytes_macros(source):
    rewritten, num_occurrences = BYTES_MACRO_RE.subn(rewrite_one_macro, source)
    if num_occurrences > 0:
        return rewritten


BYTES_MACRO_RE = re.compile(br'bytes!\(  (?P<args>  [^)]*  )  \)', re.VERBOSE)


def rewrite_one_macro(match):
    try:
        bytes = parse_bytes(split_args(match.group('args')))
        return b'b"' + b''.join(map(escape, bytes)) + b'"'
    except SkipThisRewrite:
        print('Skipped: %s' % match.group(0).decode('utf8', 'replace'))
        return match.group(0)


class SkipThisRewrite(Exception):
    pass


def split_args(args):
    previous = b''
    for arg in args.split(b','):
        if previous:
            arg = previous + b',' + arg
        if arg.count(b'"') % 2 == 0:
            yield arg
            previous = b''
        else:
            previous = arg
    if previous:
        yield previous


def parse_bytes(args):
    for arg in args:
        arg = arg.strip()
        if (arg.startswith(b'"') and arg.endswith(b'"')) or (
                arg.startswith(b"'") and arg.endswith(b"'")):
            # Escaped newline means something different in Rust and Python.
            if b'\\\n' in arg:
                raise SkipThisRewrite
            for byte in eval(b'u' + arg).encode('utf8'):
                yield ord(byte)
        else:
            if arg.endswith(b'u8'):
                arg = arg[:-2]
            # Assume that all Rust integer literals
            # are valid Python integer literals
            value = int(eval(arg))
            assert value <= 0xFF
            yield value


def escape(byte):
    c = chr(byte)
    escaped = {
        b'\0': br'\0',
        b'\t': br'\t',
        b'\n': br'\n',
        b'\r': br'\r',
        b'\'': b'\\\'',
        b'\\': br'\\',
    }.get(c)
    if escaped is not None:
        return escaped
    elif b' ' <= c <= b'~':
        return chr(byte)
    else:
        return ('\\x%02X' % byte).encode('ascii')


if str is not bytes:
    # Python 3.x
    ord = lambda x: x
    chr = lambda x: bytes([x])


if __name__ == '__main__':
    main()
