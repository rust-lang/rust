#!/bin/sh
set -eu

# Performs `cat` and `grep` simultaneously for `run-make` tests in the Rust CI.
#
# This program will read lines from stdin and print them to stdout immediately.
# At the same time, it will check if the input line contains the substring or
# regex specified in the command line. If any match is found, the program will
# set the exit code to 0, otherwise 1.
#
# This is written to simplify debugging runmake tests. Since `grep` swallows all
# output, when a test involving `grep` failed, it is impossible to know the
# reason just by reading the failure log. While it is possible to `tee` the
# output into another stream, it becomes pretty annoying to do this for all test
# cases.

USAGE='
cat-and-grep.sh [-v] [-e] [-i] s1 s2 s3 ... < input.txt

Prints the stdin, and exits successfully only if all of `sN` can be found in
some lines of the input.

Options:
    -v      Invert match, exits successfully only if all of `sN` cannot be found
    -e      Regex search, search using extended Regex instead of fixed string
    -i      Case insensitive search.
'

GREPPER=grep
INVERT=0
GREPFLAGS='q'
while getopts ':vieh' OPTION; do
    case "$OPTION" in
        v)
            INVERT=1
            ;;
        i)
            GREPFLAGS="i$GREPFLAGS"
            ;;
        e)
            GREPFLAGS="E$GREPFLAGS"
            ;;
        h)
            echo "$USAGE"
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if ! echo "$GREPFLAGS" | grep -q E
then
    # use F flag if there is not an E flag
    GREPFLAGS="F$GREPFLAGS"
fi

shift $((OPTIND - 1))

# use gnu version of tool if available (for bsd)
if command -v "g${GREPPER}"; then
    GREPPER="g${GREPPER}"
fi

LOG=$(mktemp -t cgrep.XXXXXX)
trap "rm -f $LOG" EXIT

printf "[[[ begin stdout ]]]\n\033[90m"
tee "$LOG"
echo >> "$LOG"   # ensure at least 1 line of output, otherwise `grep -v` may unconditionally fail.
printf "\033[0m\n[[[ end stdout ]]]\n"

HAS_ERROR=0
for MATCH in "$@"; do
    if "$GREPPER" "-$GREPFLAGS" -- "$MATCH" "$LOG"; then
        if [ "$INVERT" = 1 ]; then
            printf "\033[1;31mError: should not match: %s\033[0m\n" "$MATCH"
            HAS_ERROR=1
        fi
    else
        if [ "$INVERT" = 0 ]; then
            printf "\033[1;31mError: cannot match: %s\033[0m\n" "$MATCH"
            HAS_ERROR=1
        fi
    fi
done

exit "$HAS_ERROR"
