#!/usr/bin/env python

"""
This script creates a pile of compile-fail tests check that all the
derives have spans that point to the fields, rather than the
#[derive(...)] line.

sample usage: src/etc/generate-deriving-span-tests.py
"""

import os, datetime, stat, re

TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../test/ui/derives/'))

YEAR = datetime.datetime.now().year

TEMPLATE = """
// This file was auto-generated using 'src/etc/generate-deriving-span-tests.py'

{error_deriving}
struct Error;
{code}
fn main() {{}}
"""

ENUM_STRING = """
#[derive({traits})]
enum Enum {{
   A(
     Error {errors}
     )
}}
"""
ENUM_STRUCT_VARIANT_STRING = """
#[derive({traits})]
enum Enum {{
   A {{
     x: Error {errors}
   }}
}}
"""
STRUCT_STRING = """
#[derive({traits})]
struct Struct {{
    x: Error {errors}
}}
"""
STRUCT_TUPLE_STRING = """
#[derive({traits})]
struct Struct(
    Error {errors}
);
"""

ENUM_TUPLE, ENUM_STRUCT, STRUCT_FIELDS, STRUCT_TUPLE = range(4)

def create_test_case(type, trait, super_traits, error_count):
    string = [ENUM_STRING, ENUM_STRUCT_VARIANT_STRING, STRUCT_STRING, STRUCT_TUPLE_STRING][type]
    all_traits = ','.join([trait] + super_traits)
    super_traits = ','.join(super_traits)
    error_deriving = '#[derive(%s)]' % super_traits if super_traits else ''

    errors = '\n'.join('//~%s ERROR' % ('^' * n) for n in range(error_count))
    code = string.format(traits = all_traits, errors = errors)
    return TEMPLATE.format(year = YEAR, error_deriving=error_deriving, code = code)

def write_file(name, string):
    test_file = os.path.join(TEST_DIR, 'derives-span-%s.rs' % name)

    with open(test_file) as f:
        old_str = f.read()
        old_str_ignoring_date = re.sub(r'^// Copyright \d+',
                                        '// Copyright {year}'.format(year = YEAR), old_str)
        if old_str_ignoring_date == string:
            # if all we're doing is updating the copyright year, ignore it
            return 0

    # set write permission if file exists, so it can be changed
    if os.path.exists(test_file):
        os.chmod(test_file, stat.S_IWUSR)

    with open(test_file, 'w') as f:
        f.write(string)

    # mark file read-only
    os.chmod(test_file, stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)

    return 1


ENUM = 1
STRUCT = 2
ALL = STRUCT | ENUM

traits = {
    'Default': (STRUCT, [], 1),
    'FromPrimitive': (0, [], 0), # only works for C-like enums

    'Decodable': (0, [], 0), # FIXME: quoting gives horrible spans
    'Encodable': (0, [], 0), # FIXME: quoting gives horrible spans
}

for (trait, supers, errs) in [('Clone', [], 1),
                              ('PartialEq', [], 2),
                              ('PartialOrd', ['PartialEq'], 1),
                              ('Eq', ['PartialEq'], 1),
                              ('Ord', ['Eq', 'PartialOrd', 'PartialEq'], 1),
                              ('Debug', [], 1),
                              ('Hash', [], 1)]:
    traits[trait] = (ALL, supers, errs)

files = 0

for (trait, (types, super_traits, error_count)) in traits.items():
    mk = lambda ty: create_test_case(ty, trait, super_traits, error_count)
    if types & ENUM:
        files += write_file(trait + '-enum', mk(ENUM_TUPLE))
        files += write_file(trait + '-enum-struct-variant', mk(ENUM_STRUCT))
    if types & STRUCT:
        files += write_file(trait + '-struct', mk(STRUCT_FIELDS))
        files += write_file(trait + '-tuple-struct', mk(STRUCT_TUPLE))

print('Generated {files} deriving span test{}.'.format('s' if files != 1 else '', files = files))
