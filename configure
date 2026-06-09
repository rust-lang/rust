#!/bin/sh

script="$(dirname "$0")"/src/bootstrap/configure.py

try() {
    cmd=$1
    shift
    T=$($cmd --version 2>/dev/null)
    if [ $? -eq 0 ]; then
        exec $cmd "$script" "$@"
    fi
}

try python3 "$@"
try python2.7 "$@"
try python27 "$@"
try python2 "$@"
exec python "$script" "$@"
