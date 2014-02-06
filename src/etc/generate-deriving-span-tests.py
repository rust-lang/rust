#!/usr/bin/env python
#
# Copyright 2013 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

"""
This script creates a pile of compile-fail tests check that all the
derivings have spans that point to the fields, rather than the
#[deriving(...)] line.

sample usage: src/etc/generate-deriving-span-tests.py
"""

import sys, os, datetime, stat

TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../test/compile-fail'))

YEAR = datetime.datetime.now().year

TEMPLATE = """// Copyright {year} The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This file was auto-generated using 'src/etc/generate-keyword-span-tests.py'

#[feature(struct_variant)];
extern mod extra;

{error_deriving}
struct Error;
{code}
fn main() {{}}
"""

ENUM_STRING = """
#[deriving({traits})]
enum Enum {{
   A(
     Error {errors}
     )
}}
"""
ENUM_STRUCT_VARIANT_STRING = """
#[deriving({traits})]
enum Enum {{
   A {{
     x: Error {errors}
   }}
}}
"""
STRUCT_STRING = """
#[deriving({traits})]
struct Struct {{
    x: Error {errors}
}}
"""
STRUCT_TUPLE_STRING = """
#[deriving({traits})]
struct Struct(
    Error {errors}
);
"""

ENUM_TUPLE, ENUM_STRUCT, STRUCT_FIELDS, STRUCT_TUPLE = range(4)

def create_test_case(type, trait, super_traits, number_of_errors):
    string = [ENUM_STRING, ENUM_STRUCT_VARIANT_STRING, STRUCT_STRING, STRUCT_TUPLE_STRING][type]
    all_traits = ','.join([trait] + super_traits)
    super_traits = ','.join(super_traits)
    error_deriving = '#[deriving(%s)]' % super_traits if super_traits else ''

    errors = '\n'.join('//~%s ERROR' % ('^' * n) for n in range(error_count))
    code = string.format(traits = all_traits, errors = errors)
    return TEMPLATE.format(year = YEAR, error_deriving=error_deriving, code = code)

def write_file(name, string):
    test_file = os.path.join(TEST_DIR, 'deriving-span-%s.rs' % name)

    # set write permission if file exists, so it can be changed
    if os.path.exists(test_file):
        os.chmod(test_file, stat.S_IWUSR)

    with open(test_file, 'wt') as f:
        f.write(string)

    # mark file read-only
    os.chmod(test_file, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)



ENUM = 1
STRUCT = 2
ALL = STRUCT | ENUM

traits = {
    'Zero': (STRUCT, [], 1),
    'Default': (STRUCT, [], 1),
    'FromPrimitive': (0, [], 0), # only works for C-like enums

    'Decodable': (0, [], 0), # FIXME: quoting gives horrible spans
    'Encodable': (0, [], 0), # FIXME: quoting gives horrible spans
}

for (trait, supers, errs) in [('Rand', [], 1),
                              ('Clone', [], 1), ('DeepClone', ['Clone'], 1),
                              ('Eq', [], 2), ('Ord', [], 8),
                              ('TotalEq', [], 1), ('TotalOrd', ['TotalEq'], 1),
                              ('Show', [], 1)]:
    traits[trait] = (ALL, supers, errs)

for (trait, (types, super_traits, error_count)) in traits.items():
    mk = lambda ty: create_test_case(ty, trait, super_traits, error_count)
    if types & ENUM:
        write_file(trait + '-enum', mk(ENUM_TUPLE))
        write_file(trait + '-enum-struct-variant', mk(ENUM_STRUCT))
    if types & STRUCT:
        write_file(trait + '-struct', mk(STRUCT_FIELDS))
        write_file(trait + '-tuple-struct', mk(STRUCT_TUPLE))
