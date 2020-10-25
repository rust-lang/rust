#!/usr/bin/env python

import sys
import six

# Support python 2 or 3
from six.moves.urllib.parse import quote

# Converts the input string into a valid URL parameter string.
print (quote(' '.join(sys.argv[1:])))
