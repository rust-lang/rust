# Copyright 2010-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import sys
import fileinput
import subprocess
import re
import os
from licenseck import *
import snapshot

err = 0
cols = 100
cr_flag = "ignore-tidy-cr"
tab_flag = "ignore-tidy-tab"
linelength_flag = "ignore-tidy-linelength"

# Be careful to support Python 2.4, 2.6, and 3.x here!
config_proc = subprocess.Popen(["git", "config", "core.autocrlf"],
                               stdout=subprocess.PIPE)
result = config_proc.communicate()[0]

true = "true".encode('utf8')
autocrlf = result.strip() == true if result is not None else False


def report_error_name_no(name, no, s):
    global err
    print("%s:%d: %s" % (name, no, s))
    err = 1


def report_err(s):
    report_error_name_no(fileinput.filename(), fileinput.filelineno(), s)


def report_warn(s):
    print("%s:%d: %s" % (fileinput.filename(),
                         fileinput.filelineno(),
                         s))


def do_license_check(name, contents):
    if not check_license(name, contents):
        report_error_name_no(name, 1, "incorrect license")

current_name = ""
current_contents = ""
check_tab = True
check_cr = True
check_linelength = True

if len(sys.argv) < 2:
    print "usage: tidy.py <src-dir>"
    sys.exit(1)

src_dir = sys.argv[1]

try:
    count_rs = 0
    count_py = 0
    count_js = 0
    count_sh = 0
    count_pl = 0
    count_c = 0
    count_h = 0
    count_other = 0

    count_lines = 0
    count_non_blank_lines = 0

    def update_counts(current_name):
        global count_rs
        global count_py
        global count_js
        global count_sh
        global count_pl
        global count_c
        global count_h
        global count_other

        if current_name.endswith(".rs"):
            count_rs += 1
        if current_name.endswith(".py"):
            count_py += 1
        if current_name.endswith(".js"):
            count_js += 1
        if current_name.endswith(".sh"):
            count_sh += 1
        if current_name.endswith(".pl"):
            count_pl += 1
        if current_name.endswith(".c"):
            count_c += 1
        if current_name.endswith(".h"):
            count_h += 1

    all_paths = set()

    for (dirpath, dirnames, filenames) in os.walk(src_dir):

        # Skip some third-party directories
        if "src/jemalloc" in dirpath: continue
        if "src/llvm" in dirpath: continue
        if "src/gyp" in dirpath: continue
        if "src/libbacktrace" in dirpath: continue
        if "src/compiler-rt" in dirpath: continue
        if "src/rt/hoedown" in dirpath: continue
        if "src/rustllvm" in dirpath: continue
        if "src/rt/valgrind" in dirpath: continue
        if "src/rt/msvc" in dirpath: continue
        if "src/rust-installer" in dirpath: continue

        def interesting_file(f):
            if "miniz.c" in f \
            or "jquery" in f \
            or "rust_android_dummy" in f:
                return False

            if f.endswith(".rs") \
            or f.endswith(".py") \
            or f.endswith(".js") \
            or f.endswith(".sh") \
            or f.endswith(".pl") \
            or f.endswith(".c") \
            or f.endswith(".h") :
                return True
            else:
                return False

        file_names = [os.path.join(dirpath, f) for f in filenames
                      if interesting_file(f)
                      and not f.endswith("_gen.rs")
                      and not ".#" is f]

        if not file_names:
            continue

        for line in fileinput.input(file_names,
                                openhook=fileinput.hook_encoded("utf-8")):

            filename = fileinput.filename()

            if "tidy.py" not in filename:
                if "TODO" in line:
                    report_err("TODO is deprecated; use FIXME")
                match = re.match(r'^.*/(\*|/!?)\s*XXX', line)
                if match:
                    report_err("XXX is no longer necessary, use FIXME")
                match = re.match(r'^.*//\s*(NOTE.*)$', line)
                if match and "TRAVIS" not in os.environ:
                    m = match.group(1)
                    if "snap" in m.lower():
                        report_warn(match.group(1))
                match = re.match(r'^.*//\s*SNAP\s+(\w+)', line)
                if match:
                    hsh = match.group(1)
                    date, rev = snapshot.curr_snapshot_rev()
                    if not hsh.startswith(rev):
                        report_err("snapshot out of date (" + date
                            + "): " + line)
                else:
                    if "SNAP" in line:
                        report_warn("unmatched SNAP line: " + line)

            if cr_flag in line:
                check_cr = False
            if tab_flag in line:
                check_tab = False
            if linelength_flag in line:
                check_linelength = False

            if check_tab and ('\t' in line and
                              "Makefile" not in filename):
                report_err("tab character")
            if check_cr and not autocrlf and '\r' in line:
                report_err("CR character")
            if line.endswith(" \n") or line.endswith("\t\n"):
                report_err("trailing whitespace")
            line_len = len(line)-2 if autocrlf else len(line)-1

            if check_linelength and line_len > cols:
                report_err("line longer than %d chars" % cols)

            if fileinput.isfirstline():
                # This happens at the end of each file except the last.
                if current_name != "":
                    update_counts(current_name)
                    assert len(current_contents) > 0
                    do_license_check(current_name, current_contents)

                current_name = filename
                current_contents = ""
                check_cr = True
                check_tab = True
                check_linelength = True

            # Put a reasonable limit on the amount of header data we use for
            # the licenseck
            if len(current_contents) < 1000:
                current_contents += line

            count_lines += 1
            if line.strip():
                count_non_blank_lines += 1

    if current_name != "":
        update_counts(current_name)
        assert len(current_contents) > 0
        do_license_check(current_name, current_contents)

except UnicodeDecodeError as e:
    report_err("UTF-8 decoding error " + str(e))

print
print "* linted .rs files: " + str(count_rs)
print "* linted .py files: " + str(count_py)
print "* linted .js files: " + str(count_js)
print "* linted .sh files: " + str(count_sh)
print "* linted .pl files: " + str(count_pl)
print "* linted .c files: " + str(count_c)
print "* linted .h files: " + str(count_h)
print "* other linted files: " + str(count_other)
print "* total lines of code: " + str(count_lines)
print "* total non-blank lines of code: " + str(count_non_blank_lines)
print

sys.exit(err)
