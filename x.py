# The beginning of this script is both valid shell and valid python, such that
# the script starts with the shell and is reexecuted with the right python.
# This works because shells only execute a line at a time.
# Thanks to `./mach` from servo for the idea!
''':' && {
exists() { command -v "$1" >/dev/null 2>&1; }
if exists python3; then
    exec python3 "$0" "$@"
elif exists python; then
    exec python "$0" "$@"
else
    exec python2 "$0" "$@"
fi
}
'''

# This file is only a "symlink" to bootstrap.py, all logic should go there.

import os
import sys
rust_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))

import bootstrap
bootstrap.main()
