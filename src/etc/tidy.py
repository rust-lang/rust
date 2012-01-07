#!/usr/bin/env python

import sys, fileinput, subprocess

err=0
cols=78

# Be careful to support Python 2.4, 2.6, and 3.x here!
config_proc=subprocess.Popen([ "git", "config", "core.autocrlf" ],
                             stdout=subprocess.PIPE)
result=config_proc.communicate()[0]

true="true".encode('utf8')
autocrlf=result.strip() == true if result is not None else False

def report_err(s):
    global err
    print("%s:%d: %s" % (fileinput.filename(), fileinput.filelineno(), s))
    err=1

try:
    for line in fileinput.input(openhook=fileinput.hook_encoded("utf-8")):
        if (line.find('\t') != -1 and
            fileinput.filename().find("Makefile") == -1):
            report_err("tab character")
        if not autocrlf and line.find('\r') != -1:
            report_err("CR character")
        if line.endswith(" \n") or line.endswith("\t\n"):
            report_err("trailing whitespace")
        line_len = len(line)-2 if autocrlf else len(line)-1
        if line_len > cols:
            report_err("line longer than %d chars" % cols)
except UnicodeDecodeError, e:
    report_err("UTF-8 decoding error " + str(e))


sys.exit(err)

