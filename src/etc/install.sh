#!/bin/sh
# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

msg() {
    echo "install: $1"
}

step_msg() {
    msg
    msg "$1"
    msg
}

warn() {
    echo "install: WARNING: $1"
}

err() {
    echo "install: error: $1"
    exit 1
}

need_ok() {
    if [ $? -ne 0 ]
    then
        err "$1"
    fi
}

putvar() {
    local T
    eval T=\$$1
    eval TLEN=\${#$1}
    if [ $TLEN -gt 35 ]
    then
        printf "install: %-20s := %.35s ...\n" $1 "$T"
    else
        printf "install: %-20s := %s %s\n" $1 "$T" "$2"
    fi
    printf "%-20s := %s\n" $1 "$T" >>config.tmp
}

valopt() {
    VAL_OPTIONS="$VAL_OPTIONS $1"

    local OP=$1
    local DEFAULT=$2
    shift
    shift
    local DOC="$*"
    if [ $HELP -eq 0 ]
    then
        local UOP=$(echo $OP | tr '[:lower:]' '[:upper:]' | tr '\-' '\_')
        local V="CFG_${UOP}"
        eval $V="$DEFAULT"
        for arg in $CFG_ARGS
        do
            if echo "$arg" | grep -q -- "--$OP="
            then
                val=$(echo "$arg" | cut -f2 -d=)
                eval $V=$val
            fi
        done
        putvar $V
    else
        if [ -z "$DEFAULT" ]
        then
            DEFAULT="<none>"
        fi
        OP="${OP}=[${DEFAULT}]"
        printf "    --%-30s %s\n" "$OP" "$DOC"
    fi
}

opt() {
    BOOL_OPTIONS="$BOOL_OPTIONS $1"

    local OP=$1
    local DEFAULT=$2
    shift
    shift
    local DOC="$*"
    local FLAG=""

    if [ $DEFAULT -eq 0 ]
    then
        FLAG="enable"
    else
        FLAG="disable"
        DOC="don't $DOC"
    fi

    if [ $HELP -eq 0 ]
    then
        for arg in $CFG_ARGS
        do
            if [ "$arg" = "--${FLAG}-${OP}" ]
            then
                OP=$(echo $OP | tr 'a-z-' 'A-Z_')
                FLAG=$(echo $FLAG | tr 'a-z' 'A-Z')
                local V="CFG_${FLAG}_${OP}"
                eval $V=1
                putvar $V
            fi
        done
    else
        if [ ! -z "$META" ]
        then
            OP="$OP=<$META>"
        fi
        printf "    --%-30s %s\n" "$FLAG-$OP" "$DOC"
     fi
}

flag() {
    BOOL_OPTIONS="$BOOL_OPTIONS $1"

    local OP=$1
    shift
    local DOC="$*"

    if [ $HELP -eq 0 ]
    then
        for arg in $CFG_ARGS
        do
            if [ "$arg" = "--${OP}" ]
            then
                OP=$(echo $OP | tr 'a-z-' 'A-Z_')
                local V="CFG_${OP}"
                eval $V=1
                putvar $V
            fi
        done
    else
        if [ ! -z "$META" ]
        then
            OP="$OP=<$META>"
        fi
        printf "    --%-30s %s\n" "$OP" "$DOC"
     fi
}

validate_opt () {
    for arg in $CFG_ARGS
    do
        isArgValid=0
        for option in $BOOL_OPTIONS
        do
            if test --disable-$option = $arg
            then
                isArgValid=1
            fi
            if test --enable-$option = $arg
            then
                isArgValid=1
            fi
            if test --$option = $arg
            then
                isArgValid=1
            fi
        done
        for option in $VAL_OPTIONS
        do
            if echo "$arg" | grep -q -- "--$option="
            then
                isArgValid=1
            fi
        done
        if [ "$arg" = "--help" ]
        then
            echo
            echo "No more help available for Configure options,"
            echo "check the Wiki or join our IRC channel"
            break
        else
            if test $isArgValid -eq 0
            then
                err "Option '$arg' is not recognized"
            fi
        fi
    done
}

CFG_SRC_DIR="$(cd $(dirname $0) && pwd)/"
CFG_SELF="$0"
CFG_ARGS="$@"

HELP=0
if [ "$1" = "--help" ]
then
    HELP=1
    shift
    echo
    echo "Usage: $CFG_SELF [options]"
    echo
    echo "Options:"
    echo
else
    step_msg "processing $CFG_SELF args"
fi

OPTIONS=""
BOOL_OPTIONS=""
VAL_OPTIONS=""

flag uninstall "only uninstall from the installation prefix"
valopt prefix "/usr/local" "set installation prefix"

if [ $HELP -eq 1 ]
then
    echo
    exit 0
fi

step_msg "validating $CFG_SELF args"
validate_opt

# Sanity check: can we can write to the destination?
touch "${CFG_PREFIX}/lib/rust-install-probe" 2> /dev/null
if [ $? -ne 0 ]
then
    err "can't write to destination. try again with 'sudo'."
fi
rm -r "${CFG_PREFIX}/lib/rust-install-probe"
need_ok "failed to remove install probe"

# Sanity check: can we run these binaries?
"${CFG_SRC_DIR}/bin/rustc" --version > /dev/null
need_ok "can't run these binaries on this platform"

# First, uninstall from the installation prefix
# FIXME: Hardcoded 'rustlib' ignores CFG_RUSTLIBDIR
if [ -f "${CFG_PREFIX}/lib/rustlib/manifest" ]
then
    while read p; do
        msg "uninstall ${CFG_PREFIX}/$p"
        rm "${CFG_PREFIX}/$p"
        need_ok "failed to remove file"
    done < "${CFG_PREFIX}/lib/rustlib/manifest"

    # Remove 'rustlib' directory
    msg "uninstall ${CFG_PREFIX}/lib/rustlib"
    rm -r "${CFG_PREFIX}/lib/rustlib"
    need_ok "failed to remove rustlib"
fi

# If we're only uninstalling then exit
if [ -n "${CFG_UNINSTALL}" ]
then
    exit 0
fi

# Iterate through the new manifest and install files
while read p; do

    umask 022 && mkdir -p "${CFG_PREFIX}/$(dirname $p)"
    need_ok "directory creation failed"

    msg "${CFG_PREFIX}/$p"
    if echo "$p" | grep "/bin/" > /dev/null
    then
        install -m755 "${CFG_SRC_DIR}/$p" "${CFG_PREFIX}/$p"
    else
        install -m644 "${CFG_SRC_DIR}/$p" "${CFG_PREFIX}/$p"
    fi
    need_ok "file creation failed"

# The manifest lists all files to install
done < "${CFG_SRC_DIR}/lib/rustlib/manifest"

echo
echo "    Rust is ready to roll."
echo


