#!/usr/bin/env python
#
# Copyright 2011-2013 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This digests UnicodeData.txt and DerivedCoreProperties.txt and emits rust
# code covering the core properties. Since this is a pretty rare event we
# just store this out-of-line and check the unicode.rs file into git.
#
# The emitted code is "the minimum we think is necessary for libstd", that
# is, to support basic operations of the compiler and "most nontrivial rust
# programs". It is not meant to be a complete implementation of unicode.
# For that we recommend you use a proper binding to libicu.

import fileinput, re, os, sys, operator


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
    upperlower = {}
    lowerupper = {}
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
         old, iso, upcase, lowcase, titlecase ] = fields

        code_org = code
        code     = int(code, 16)

        # generate char to char direct common and simple conversions
        # uppercase to lowercase
        if gencat == "Lu" and lowcase != "" and code_org != lowcase:
            upperlower[code] = int(lowcase, 16)

        # lowercase to uppercase
        if gencat == "Ll" and upcase != "" and code_org != upcase:
            lowerupper[code] = int(upcase, 16)

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

    return (canon_decomp, compat_decomp, gencats, combines, lowerupper, upperlower)

def load_properties(f, interestingprops):
    fetch(f)
    props = {}
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
        if prop not in props:
            props[prop] = []
        props[prop].append((d_lo, d_hi))
    return props

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
    use slice::ImmutableVector;
    use option::None;
    r.bsearch(|&(lo,hi)| {
        if lo <= c && c <= hi { Equal }
        else if hi < c { Less }
        else { Greater }
    }) != None
}\n
""");

def emit_property_module(f, mod, tbl):
    f.write("pub mod %s {\n" % mod)
    keys = tbl.keys()
    keys.sort()

    for cat in keys:
        if cat not in ["Nd", "Nl", "No", "Cc",
            "XID_Start", "XID_Continue", "Alphabetic",
            "Lowercase", "Uppercase", "White_Space"]:
            continue
        f.write("    static %s_table : &'static [(char,char)] = &[\n" % cat)
        ix = 0
        for pair in tbl[cat]:
            f.write(ch_prefix(ix))
            f.write("(%s, %s)" % (escape_char(pair[0]), escape_char(pair[1])))
            ix += 1
        f.write("\n    ];\n\n")

        f.write("    pub fn %s(c: char) -> bool {\n" % cat)
        f.write("        super::bsearch_range_table(c, %s_table)\n" % cat)
        f.write("    }\n\n")
    f.write("}\n\n")


def emit_conversions_module(f, lowerupper, upperlower):
    f.write("pub mod conversions {")
    f.write("""
    use cmp::{Equal, Less, Greater};
    use slice::ImmutableVector;
    use tuple::Tuple2;
    use option::{Option, Some, None};

    pub fn to_lower(c: char) -> char {
        match bsearch_case_table(c, LuLl_table) {
          None        => c,
          Some(index) => LuLl_table[index].val1()
        }
    }

    pub fn to_upper(c: char) -> char {
        match bsearch_case_table(c, LlLu_table) {
            None        => c,
            Some(index) => LlLu_table[index].val1()
        }
    }

    fn bsearch_case_table(c: char, table: &'static [(char, char)]) -> Option<uint> {
        table.bsearch(|&(key, _)| {
            if c == key { Equal }
            else if key < c { Less }
            else { Greater }
        })
    }

""");
    emit_caseconversion_table(f, "LuLl", upperlower)
    emit_caseconversion_table(f, "LlLu", lowerupper)
    f.write("}\n")

def emit_caseconversion_table(f, name, table):
    f.write("    static %s_table : &'static [(char, char)] = &[\n" % name)
    sorted_table = sorted(table.iteritems(), key=operator.itemgetter(0))
    ix = 0
    for key, value in sorted_table:
        f.write(ch_prefix(ix))
        f.write("(%s, %s)" % (escape_char(key), escape_char(value)))
        ix += 1
    f.write("\n    ];\n\n")

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

def emit_core_norm_module(f, canon, compat):
    canon_keys = canon.keys()
    canon_keys.sort()

    compat_keys = compat.keys()
    compat_keys.sort()
    f.write("pub mod normalization {\n");
    f.write("    use option::Option;\n");
    f.write("    use option::{Some, None};\n");
    f.write("    use slice::ImmutableVector;\n");
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

    f.write("""
    pub fn decompose_canonical(c: char, i: |char|) { d(c, i, false); }

    pub fn decompose_compatible(c: char, i: |char|) { d(c, i, true); }

    fn d(c: char, i: |char|, k: bool) {
        use iter::Iterator;

        // 7-bit ASCII never decomposes
        if c <= '\\x7f' { i(c); return; }

        // Perform decomposition for Hangul
        if (c as u32) >= S_BASE && (c as u32) < (S_BASE + S_COUNT) {
            decompose_hangul(c, i);
            return;
        }

        // First check the canonical decompositions
        match bsearch_table(c, canonical_table) {
            Some(canon) => {
                for x in canon.iter() {
                    d(*x, |b| i(b), k);
                }
                return;
            }
            None => ()
        }

        // Bottom out if we're not doing compat.
        if !k { i(c); return; }

        // Then check the compatibility decompositions
        match bsearch_table(c, compatibility_table) {
            Some(compat) => {
                for x in compat.iter() {
                    d(*x, |b| i(b), k);
                }
                return;
            }
            None => ()
        }

        // Finally bottom out.
        i(c);
    }

    // Constants from Unicode 6.2.0 Section 3.12 Conjoining Jamo Behavior
    static S_BASE: u32 = 0xAC00;
    static L_BASE: u32 = 0x1100;
    static V_BASE: u32 = 0x1161;
    static T_BASE: u32 = 0x11A7;
    static L_COUNT: u32 = 19;
    static V_COUNT: u32 = 21;
    static T_COUNT: u32 = 28;
    static N_COUNT: u32 = (V_COUNT * T_COUNT);
    static S_COUNT: u32 = (L_COUNT * N_COUNT);

    // Decompose a precomposed Hangul syllable
    fn decompose_hangul(s: char, f: |char|) {
        use cast::transmute;

        let si = s as u32 - S_BASE;

        let li = si / N_COUNT;
        unsafe {
            f(transmute(L_BASE + li));

            let vi = (si % N_COUNT) / T_COUNT;
            f(transmute(V_BASE + vi));

            let ti = si % T_COUNT;
            if ti > 0 {
                f(transmute(T_BASE + ti));
            }
        }
    }
}

""")

def emit_std_norm_module(f, combine):
    f.write("pub mod normalization {\n");
    f.write("    use option::{Some, None};\n");
    f.write("    use slice::ImmutableVector;\n");

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

    f.write("    static combining_class_table : &'static [(char, char, u8)] = &[\n")
    ix = 0
    for pair in combine:
        f.write(ch_prefix(ix))
        f.write("(%s, %s, %s)" % (escape_char(pair[0]), escape_char(pair[1]), pair[2]))
        ix += 1
    f.write("\n    ];\n\n")

    f.write("    pub fn canonical_combining_class(c: char) -> u8 {\n"
        + "        bsearch_range_value_table(c, combining_class_table)\n"
        + "    }\n")
    f.write("}\n")


preamble = '''// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NOTE: The following code was generated by "src/etc/unicode.py", do not edit directly

#![allow(missing_doc, non_uppercase_statics)]

'''

(canon_decomp, compat_decomp, gencats,
 combines, lowerupper, upperlower) = load_unicode_data("UnicodeData.txt")

def gen_core_unicode():
    r = "core_unicode.rs"
    if os.path.exists(r):
        os.remove(r);
    with open(r, "w") as rf:
        # Preamble
        rf.write(preamble)

        emit_bsearch_range_table(rf);
        emit_property_module(rf, "general_category", gencats)

        emit_core_norm_module(rf, canon_decomp, compat_decomp)

        derived = load_properties("DerivedCoreProperties.txt",
                ["XID_Start", "XID_Continue", "Alphabetic", "Lowercase", "Uppercase"])

        emit_property_module(rf, "derived_property", derived)

        props = load_properties("PropList.txt", ["White_Space"])
        emit_property_module(rf, "property", props)
        emit_conversions_module(rf, lowerupper, upperlower)

def gen_std_unicode():
    r = "std_unicode.rs"
    if os.path.exists(r):
        os.remove(r);
    with open(r, "w") as rf:
        # Preamble
        rf.write(preamble)
        emit_std_norm_module(rf, combines)

gen_core_unicode()
gen_std_unicode()
