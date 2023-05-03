#!/bin/sh

# Modern Linux and macOS systems commonly only have a thing called `python3` and
# not `python`, while Windows commonly does not have `python3`, so we cannot
# directly use python in the x.py shebang and have it consistently work. Instead we
# have a shell script to look for a python to run x.py.

set -eu

# syntax check
sh -n $0

realpath() {
    if [ -d "$1" ]; then
        CDPATH='' command cd "$1" && pwd -P
    else
        echo "$(realpath "$(dirname "$1")")/$(basename "$1")"
    fi
}

xpy=$(dirname "$(realpath "$0")")/x.py

# On Windows, `py -3` sometimes works. We need to try it first because `python3`
# sometimes tries to launch the app store on Windows.
for SEARCH_PYTHON in py python3 python python2; do
    if python=$(command -v $SEARCH_PYTHON) && [ -x "$python" ]; then
        if [ $SEARCH_PYTHON = py ]; then
            extra_arg="-3"
        else
            extra_arg=""
        fi
        exec "$python" $extra_arg "$xpy" "$@"
    fi
done

python=$(bash -c "compgen -c python" | grep '^python[2-3]\.[0-9]\+$' | head -n1)
if ! [ "$python" = "" ]; then
    exec "$python" "$xpy" "$@"
fi

echo "$0: error: did not find python installed" >&2
exit 1
