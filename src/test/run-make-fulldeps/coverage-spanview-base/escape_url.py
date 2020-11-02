#!/usr/bin/env python

import sys

# Support python 2 or 3
try:
    from urllib.parse import quote
except ImportError:
    from urllib import quote

# Converts the input string into a valid URL parameter string.
print (quote(' '.join(sys.argv[1:])))
