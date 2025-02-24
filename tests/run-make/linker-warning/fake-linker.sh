#!/bin/sh

code=0
while ! [ $# = 0 ]; do
    case "$1" in
        run_make_info) echo "foo"
            ;;
        run_make_warn) echo "warning: bar" >&2
            ;;
        run_make_error) echo "error: baz" >&2; code=1
            ;;
        *) ;;   # rustc passes lots of args we don't care about
    esac
    shift
done

exit $code
