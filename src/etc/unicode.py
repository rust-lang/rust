#!/usr/bin/env python

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


def load_general_categories(f):
    fetch(f)
    gencats = {}
    curr_cat = ""
    c_lo = 0
    c_hi = 0
    for line in fileinput.input(f):
        fields = line.split(";")
        if len(fields) != 15:
            continue
        [code, name, gencat, combine, bidi,
         decomp, deci, digit, num, mirror,
         old, iso, upcase, lowcsae, titlecase ] = fields

        code = int(code, 16)

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
    return gencats


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

def emit_rust_module(f, mod, tbl):
    f.write("mod %s {\n" % mod)
    keys = tbl.keys()
    keys.sort()
    for cat in keys:
        f.write("    pure fn %s(c: char) -> bool {\n" % cat)
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


def emit_cpp_module(f, mod, tbl):
    keys = tbl.keys()
    keys.sort()

    for cat in keys:

        singles = []
        ranges = []

        for pair in tbl[cat]:
            if pair[0] == pair[1]:
                singles.append(pair[0])
            else:
                ranges.append(pair)

        f.write("bool %s_%s(unsigned c) {\n" % (mod, cat))
        for pair in ranges:
            f.write("    if (0x%x <= c && c <= 0x%x) { return true; }\n"
                    % pair)
        if len(singles) > 0:
            f.write("    switch (c) {\n");
            for single in singles:
                f.write("      case 0x%x:\n" % single)
            f.write("        return true;\n");
            f.write("      default:\n");
            f.write("        return false;\n");
            f.write("    }\n")
        f.write("return false;\n")
        f.write("}\n\n")


def emit_module(rf, cf, mod, tbl):
    emit_rust_module(rf, mod, tbl)
    emit_cpp_module(cf, mod, tbl)

r = "unicode.rs"
c = "unicode.cpp"
for i in [r, c]:
    if os.path.exists(i):
        os.remove(i);

rf = open(r, "w")
cf = open(c, "w")

emit_module(rf, cf, "general_category",
            load_general_categories("UnicodeData.txt"))

emit_module(rf, cf, "derived_property",
            load_derived_core_properties("DerivedCoreProperties.txt"))
