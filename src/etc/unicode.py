#!/usr/bin/env python
# xfail-license

# This digests UnicodeData.txt and DerivedCoreProperties.txt and emits rust
# code covering the core properties. Since this is a pretty rare event we
# just store this out-of-line and check the unicode.rs file into git.
#
# The emitted code is "the minimum we think is necessary for libcore", that
# is, to support basic operations of the compiler and "most nontrivial rust
# programs". It is not meant to be a complete implementation of unicode.
# For that we recommend you use a proper binding to libicu.

import fileinput, re, os, sys


def fetch(f):
    if not os.path.exists(f):
        os.system("curl -O http://www.unicode.org/Public/UNIDATA/%s"
                  % f)

    if not os.path.exists(f):
        sys.stderr.write("cannot load %s" % f)
        exit(1)


def load_unicode_data(f):
    fetch(f)
    gencats = {}
    combines = []
    canon_decomp = {}
    compat_decomp = {}
    curr_cat = ""
    curr_combine = ""
    c_lo = 0
    c_hi = 0
    com_lo = 0
    com_hi = 0
    for line in fileinput.input(f):
        fields = line.split(";")
        if len(fields) != 15:
            continue
        [code, name, gencat, combine, bidi,
         decomp, deci, digit, num, mirror,
         old, iso, upcase, lowcsae, titlecase ] = fields

        code = int(code, 16)

        if decomp != "":
            if decomp.startswith('<'):
                seq = []
                for i in decomp.split()[1:]:
                    seq.append(int(i, 16))
                compat_decomp[code] = seq
            else:
                seq = []
                for i in decomp.split():
                    seq.append(int(i, 16))
                canon_decomp[code] = seq

        if curr_cat == "":
            curr_cat = gencat
            c_lo = code
            c_hi = code

        if curr_cat == gencat:
            c_hi = code
        else:
            if curr_cat not in gencats:
                gencats[curr_cat] = []

            gencats[curr_cat].append((c_lo, c_hi))
            curr_cat = gencat
            c_lo = code
            c_hi = code

        if curr_combine == "":
            curr_combine = combine
            com_lo = code
            com_hi = code

        if curr_combine == combine:
            com_hi = code
        else:
            if curr_combine != "0":
                combines.append((com_lo, com_hi, curr_combine))
            curr_combine = combine
            com_lo = code
            com_hi = code

    return (canon_decomp, compat_decomp, gencats, combines)


def load_derived_core_properties(f):
    fetch(f)
    derivedprops = {}
    interestingprops = ["XID_Start", "XID_Continue", "Alphabetic"]
    re1 = re.compile("^([0-9A-F]+) +; (\w+)")
    re2 = re.compile("^([0-9A-F]+)\.\.([0-9A-F]+) +; (\w+)")

    for line in fileinput.input(f):
        prop = None
        d_lo = 0
        d_hi = 0
        m = re1.match(line)
        if m:
            d_lo = m.group(1)
            d_hi = m.group(1)
            prop = m.group(2)
        else:
            m = re2.match(line)
            if m:
                d_lo = m.group(1)
                d_hi = m.group(2)
                prop = m.group(3)
            else:
                continue
        if prop not in interestingprops:
            continue
        d_lo = int(d_lo, 16)
        d_hi = int(d_hi, 16)
        if prop not in derivedprops:
            derivedprops[prop] = []
        derivedprops[prop].append((d_lo, d_hi))
    return derivedprops

def escape_char(c):
    if c <= 0xff:
        return "'\\x%2.2x'" % c
    if c <= 0xffff:
        return "'\\u%4.4x'" % c
    return "'\\U%8.8x'" % c

def ch_prefix(ix):
    if ix == 0:
        return "        "
    if ix % 2 == 0:
        return ",\n        "
    else:
        return ", "

def emit_bsearch_range_table(f):
    f.write("""
    fn bsearch_range_table(c: char, r: &'static [(char,char)]) -> bool {
        use cmp::{Equal, Less, Greater};
        use vec::ImmutableVector;
        use option::None;
        (do r.bsearch |&(lo,hi)| {
            if lo <= c && c <= hi { Equal }
            else if hi < c { Less }
            else { Greater }
        }) != None
    }\n\n
""");

def emit_property_module(f, mod, tbl):
    f.write("pub mod %s {\n" % mod)
    keys = tbl.keys()
    keys.sort()
    emit_bsearch_range_table(f);
    for cat in keys:
        f.write("    static %s_table : &'static [(char,char)] = &[\n" % cat)
        ix = 0
        for pair in tbl[cat]:
            f.write(ch_prefix(ix))
            f.write("(%s, %s)" % (escape_char(pair[0]), escape_char(pair[1])))
            ix += 1
        f.write("\n    ];\n\n")

        f.write("    pub fn %s(c: char) -> bool {\n" % cat)
        f.write("        bsearch_range_table(c, %s_table)\n" % cat)
        f.write("    }\n\n")
    f.write("}\n")


def emit_property_module_old(f, mod, tbl):
    f.write("mod %s {\n" % mod)
    keys = tbl.keys()
    keys.sort()
    for cat in keys:
        f.write("    fn %s(c: char) -> bool {\n" % cat)
        f.write("        ret alt c {\n")
        prefix = ' '
        for pair in tbl[cat]:
            if pair[0] == pair[1]:
                f.write("            %c %s\n" %
                        (prefix, escape_char(pair[0])))
            else:
                f.write("            %c %s to %s\n" %
                        (prefix,
                         escape_char(pair[0]),
                         escape_char(pair[1])))
            prefix = '|'
        f.write("              { true }\n")
        f.write("            _ { false }\n")
        f.write("        };\n")
        f.write("    }\n\n")
    f.write("}\n")

def format_table_content(f, content, indent):
    line = " "*indent
    first = True
    for chunk in content.split(","):
        if len(line) + len(chunk) < 98:
            if first:
                line += chunk
            else:
                line += ", " + chunk
            first = False
        else:
            f.write(line + ",\n")
            line = " "*indent + chunk
    f.write(line)

def emit_decomp_module(f, canon, compat, combine):
    canon_keys = canon.keys()
    canon_keys.sort()

    compat_keys = compat.keys()
    compat_keys.sort()
    f.write("pub mod decompose {\n");
    f.write("    use option::Option;\n");
    f.write("    use option::{Some, None};\n");
    f.write("    use vec::ImmutableVector;\n");
    f.write("""
    fn bsearch_table(c: char, r: &'static [(char, &'static [char])]) -> Option<&'static [char]> {
        use cmp::{Equal, Less, Greater};
        match r.bsearch(|&(val, _)| {
            if c == val { Equal }
            else if val < c { Less }
            else { Greater }
        }) {
            Some(idx) => {
                let (_, result) = r[idx];
                Some(result)
            }
            None => None
        }
    }\n
""")

    f.write("""
    fn bsearch_range_value_table(c: char, r: &'static [(char, char, u8)]) -> u8 {
        use cmp::{Equal, Less, Greater};
        match r.bsearch(|&(lo, hi, _)| {
            if lo <= c && c <= hi { Equal }
            else if hi < c { Less }
            else { Greater }
        }) {
            Some(idx) => {
                let (_, _, result) = r[idx];
                result
            }
            None => 0
        }
    }\n\n
""")

    f.write("    // Canonical decompositions\n")
    f.write("    static canonical_table : &'static [(char, &'static [char])] = &[\n")
    data = ""
    first = True
    for char in canon_keys:
        if not first:
            data += ","
        first = False
        data += "(%s,&[" % escape_char(char)
        first2 = True
        for d in canon[char]:
            if not first2:
                data += ","
            first2 = False
            data += escape_char(d)
        data += "])"
    format_table_content(f, data, 8)
    f.write("\n    ];\n\n")

    f.write("    // Compatibility decompositions\n")
    f.write("    static compatibility_table : &'static [(char, &'static [char])] = &[\n")
    data = ""
    first = True
    for char in compat_keys:
        if not first:
            data += ","
        first = False
        data += "(%s,&[" % escape_char(char)
        first2 = True
        for d in compat[char]:
            if not first2:
                data += ","
            first2 = False
            data += escape_char(d)
        data += "])"
    format_table_content(f, data, 8)
    f.write("\n    ];\n\n")

    f.write("    static combining_class_table : &'static [(char, char, u8)] = &[\n")
    ix = 0
    for pair in combine:
        f.write(ch_prefix(ix))
        f.write("(%s, %s, %s)" % (escape_char(pair[0]), escape_char(pair[1]), pair[2]))
        ix += 1
    f.write("\n    ];\n")

    f.write("    pub fn canonical(c: char, i: &fn(char)) "
        + "{ d(c, i, false); }\n\n")
    f.write("    pub fn compatibility(c: char, i: &fn(char)) "
            +"{ d(c, i, true); }\n\n")
    f.write("    pub fn canonical_combining_class(c: char) -> u8 {\n"
        + "        bsearch_range_value_table(c, combining_class_table)\n"
        + "    }\n\n")
    f.write("    fn d(c: char, i: &fn(char), k: bool) {\n")
    f.write("        use iterator::Iterator;\n");

    f.write("        if c <= '\\x7f' { i(c); return; }\n")

    # First check the canonical decompositions
    f.write("""
        match bsearch_table(c, canonical_table) {
            Some(canon) => {
                for x in canon.iter() {
                    d(*x, |b| i(b), k);
                }
                return;
            }
            None => ()
        }\n\n""")

    # Bottom out if we're not doing compat.
    f.write("        if !k { i(c); return; }\n")

    # Then check the compatibility decompositions
    f.write("""
        match bsearch_table(c, compatibility_table) {
            Some(compat) => {
                for x in compat.iter() {
                    d(*x, |b| i(b), k);
                }
                return;
            }
            None => ()
        }\n\n""")

    # Finally bottom out.
    f.write("        i(c);\n")
    f.write("    }\n")
    f.write("}\n\n")

r = "unicode.rs"
for i in [r]:
    if os.path.exists(i):
        os.remove(i);
rf = open(r, "w")

(canon_decomp, compat_decomp, gencats, combines) = load_unicode_data("UnicodeData.txt")

# Preamble
rf.write('''// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The following code was generated by "src/etc/unicode.py"

#[allow(missing_doc)];
#[allow(non_uppercase_statics)];

''')

emit_property_module(rf, "general_category", gencats)

emit_decomp_module(rf, canon_decomp, compat_decomp, combines)

derived = load_derived_core_properties("DerivedCoreProperties.txt")
emit_property_module(rf, "derived_property", derived)
