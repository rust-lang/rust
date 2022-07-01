#!/usr/bin/env -S -u If_this_file_fails_to_run,read_README_ENTRYPOINT.md bash
# `py` fails when using a shebang other than /usr/bin/python{,2,3}.
# Try and give a better error by embedding a comment in the shebang.

# Modern Linux and macOS systems commonly only have a thing called `python3` and
# not `python`, while Windows commonly does not have `python3`, so we cannot
# directly use python in the shebang and have it consistently work. Instead we
# embed some bash to look for a python to run the rest of the script.
#
# On Windows, `py -3` sometimes works. We need to try it first because `python3`
# sometimes tries to launch the app store on Windows.
'''':
for PYTHON in "py -3" python3 python python2; do
    if command -v $PYTHON >/dev/null; then
        exec $PYTHON "$0" "$@"
        break
    fi
done
echo "$0: error: did not find python installed" >&2
exit 1
'''

# The rest of this file is Python.
#
# This file is only a "symlink" to bootstrap.py, all logic should go there.

import os
import sys

# If this is python2, check if python3 is available and re-execute with that
# interpreter.
#
# `./x.py` would not normally benefit from this because the bash above tries
# python3 before 2, but this matters if someone ran `python x.py` and their
# system's `python` is python2.
if sys.version_info.major < 3:
    try:
        os.execvp("py", ["py", "-3"] + sys.argv)
    except OSError:
        try:
            os.execvp("python3", ["python3"] + sys.argv)
        except OSError:
            # Python 3 isn't available, fall back to python 2
            pass

rust_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))

import bootstrap
bootstrap.main()
