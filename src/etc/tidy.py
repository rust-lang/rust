# Copyright 2010-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import sys, fileinput, subprocess, re
from licenseck import *
import snapshot

err=0
cols=100
cr_flag="ignore-tidy-cr"
tab_flag="ignore-tidy-tab"
linelength_flag="ignore-tidy-linelength"

# Be careful to support Python 2.4, 2.6, and 3.x here!
config_proc=subprocess.Popen([ "git", "config", "core.autocrlf" ],
                             stdout=subprocess.PIPE)
result=config_proc.communicate()[0]

true="true".encode('utf8')
autocrlf=result.strip() == true if result is not None else False

def report_error_name_no(name, no, s):
    global err
    print("%s:%d: %s" % (name, no, s))
    err=1

def report_err(s):
    report_error_name_no(fileinput.filename(), fileinput.filelineno(), s)

def report_warn(s):
    print("%s:%d: %s" % (fileinput.filename(),
                         fileinput.filelineno(),
                         s))

def do_license_check(name, contents):
    if not check_license(name, contents):
        report_error_name_no(name, 1, "incorrect license")


file_names = [s for s in sys.argv[1:] if (not s.endswith("_gen.rs"))
                                     and (not ".#" in s)]

current_name = ""
current_contents = ""
check_tab = True
check_cr = True
check_linelength = True


try:
    for line in fileinput.input(file_names,
                                openhook=fileinput.hook_encoded("utf-8")):

        if fileinput.filename().find("tidy.py") == -1:
            if line.find(cr_flag) != -1:
                check_cr = False
            if line.find(tab_flag) != -1:
                check_tab = False
            if line.find(linelength_flag) != -1:
                check_linelength = False
            if line.find("// XXX") != -1:
                report_err("XXX is no longer necessary, use FIXME")
            if line.find("TODO") != -1:
                report_err("TODO is deprecated; use FIXME")
            match = re.match(r'^.*//\s*(NOTE.*)$', line)
            if match:
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

        if check_tab and (line.find('\t') != -1 and
            fileinput.filename().find("Makefile") == -1):
            report_err("tab character")
        if check_cr and not autocrlf and line.find('\r') != -1:
            report_err("CR character")
        if line.endswith(" \n") or line.endswith("\t\n"):
            report_err("trailing whitespace")
        line_len = len(line)-2 if autocrlf else len(line)-1

        if check_linelength and line_len > cols:
            report_err("line longer than %d chars" % cols)

        if fileinput.isfirstline() and current_name != "":
            do_license_check(current_name, current_contents)

        if fileinput.isfirstline():
            current_name = fileinput.filename()
            current_contents = ""
            check_cr = True
            check_tab = True
            check_linelength = True

        current_contents += line

    if current_name != "":
        do_license_check(current_name, current_contents)

except UnicodeDecodeError, e:
    report_err("UTF-8 decoding error " + str(e))


sys.exit(err)
