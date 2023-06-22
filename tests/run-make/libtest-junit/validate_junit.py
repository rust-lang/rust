#!/usr/bin/env python

import sys
import xml.etree.ElementTree as ET

# Try to decode line in order to ensure it is a valid XML document
for line in sys.stdin:
    try:
        ET.fromstring(line)
    except ET.ParseError:
        print("Invalid xml: %r" % line)
        raise
