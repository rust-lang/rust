#!/usr/bin/env python

# Trivial Python script that reads lines from stdin, and checks that each line
# is a well-formed XML document.
#
# This takes advantage of the fact that Python has a built-in XML parser,
# whereas doing the same check in Rust would require us to pull in an XML
# crate just for this relatively-minor test.
#
# If you're trying to remove Python scripts from the test suite, think twice
# before removing this one. You could do so, but it's probably not worth it.

import sys
import xml.etree.ElementTree as ET

# Try to decode line in order to ensure it is a valid XML document
for line in sys.stdin:
    try:
        ET.fromstring(line)
    except ET.ParseError:
        print("Invalid xml: %r" % line)
        raise
