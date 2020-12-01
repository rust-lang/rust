#!/bin/bash

# A script to update the references for particular tests. The idea is
# that you do a run, which will generate files in the build directory
# containing the (normalized) actual output of the compiler. This
# script will then copy that output and replace the "expected output"
# files. You can then commit the changes.
#
# If you find yourself manually editing a `foo.stderr` file, you're
# doing it wrong.

if [[ "$1" == "--help" || "$1" == "-h" || "$1" == "" || "$2" == "" ]]; then
    echo "usage: $0 <build-directory> <relative-path-to-rs-files>"
    echo ""
    echo "For example:"
    echo "   $0 ../../../build/x86_64-apple-darwin/test/ui *.rs */*.rs"
fi

MYDIR=$(dirname "$0")

BUILD_DIR="$1"
shift

while [[ "$1" != "" ]]; do
    STDERR_NAME="${1/%.rs/.stderr}"
    STDOUT_NAME="${1/%.rs/.stdout}"
    FIXED_NAME="${1/%.rs/.fixed}"
    shift
    if [[ -f "$BUILD_DIR"/"$STDOUT_NAME" ]] && \
           ! (cmp -s -- "$BUILD_DIR"/"$STDOUT_NAME" "$MYDIR"/"$STDOUT_NAME"); then
        echo updating "$MYDIR"/"$STDOUT_NAME"
        cp "$BUILD_DIR"/"$STDOUT_NAME" "$MYDIR"/"$STDOUT_NAME"
        if [[ ! -s "$MYDIR"/"$STDOUT_NAME" ]]; then
            echo removing "$MYDIR"/"$STDOUT_NAME"
            rm "$MYDIR"/"$STDOUT_NAME"
        fi
    fi
    if [[ -f "$BUILD_DIR"/"$STDERR_NAME" ]] && \
           ! (cmp -s -- "$BUILD_DIR"/"$STDERR_NAME" "$MYDIR"/"$STDERR_NAME"); then
        echo updating "$MYDIR"/"$STDERR_NAME"
        cp "$BUILD_DIR"/"$STDERR_NAME" "$MYDIR"/"$STDERR_NAME"
        if [[ ! -s "$MYDIR"/"$STDERR_NAME" ]]; then
            echo removing "$MYDIR"/"$STDERR_NAME"
            rm "$MYDIR"/"$STDERR_NAME"
        fi
    fi
    if [[ -f "$BUILD_DIR"/"$FIXED_NAME" ]] && \
           ! (cmp -s -- "$BUILD_DIR"/"$FIXED_NAME" "$MYDIR"/"$FIXED_NAME"); then
        echo updating "$MYDIR"/"$FIXED_NAME"
        cp "$BUILD_DIR"/"$FIXED_NAME" "$MYDIR"/"$FIXED_NAME"
        if [[ ! -s "$MYDIR"/"$FIXED_NAME" ]]; then
            echo removing "$MYDIR"/"$FIXED_NAME"
            rm "$MYDIR"/"$FIXED_NAME"
        fi
    fi
done
