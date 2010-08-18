#!/usr/bin/python

import sys, fileinput

err=0
cols=78

def report_err(s):
    global err
    print("%s:%d: %s" % (fileinput.filename(), fileinput.filelineno(), s))
    err=1

for line in fileinput.input(openhook=fileinput.hook_encoded("utf-8")):
    if line.find('\t') != -1 and fileinput.filename().find("Makefile") == -1:
        report_err("tab character")

    if line.find('\r') != -1:
        report_err("CR character")

    if len(line)-1 > cols:
        report_err("line longer than %d chars" % cols)


sys.exit(err)

