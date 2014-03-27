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
# NB This isn't quite the same definition as in `configure`.
# just using 'lib' instead of CFG_LIBDIR_RELATIVE
valopt libdir "${CFG_PREFIX}/lib" "install libraries"
valopt mandir "${CFG_PREFIX}/share/man" "install man pages in PATH"

if [ $HELP -eq 1 ]
then
    echo
    exit 0
fi

step_msg "validating $CFG_SELF args"
validate_opt


# OK, let's get installing ...

# Sanity check: can we can write to the destination?
umask 022 && mkdir -p "${CFG_LIBDIR}"
need_ok "can't write to destination. consider 'sudo'."
touch "${CFG_LIBDIR}/rust-install-probe" 2> /dev/null
if [ $? -ne 0 ]
then
    err "can't write to destination. consider 'sudo'."
fi
rm "${CFG_LIBDIR}/rust-install-probe"
need_ok "failed to remove install probe"

# Sanity check: don't install to the directory containing the installer.
# That would surely cause chaos.
INSTALLER_DIR="$(cd $(dirname $0) && pwd)"
PREFIX_DIR="$(cd ${CFG_PREFIX} && pwd)"
if [ "${INSTALLER_DIR}" = "${PREFIX_DIR}" ]
then
    err "can't install to same directory as installer"
fi

# The file name of the manifest we're going to create during install
INSTALLED_MANIFEST="${CFG_LIBDIR}/rustlib/manifest"

# First, uninstall from the installation prefix.
# Errors are warnings - try to rm everything in the manifest even if some fail.
if [ -f "${INSTALLED_MANIFEST}" ]
then
    # Iterate through installed manifest and remove files
    while read p; do
        # The installed manifest contains absolute paths
        msg "removing $p"
        if [ -f "$p" ]
        then
            rm "$p"
            if [ $? -ne 0 ]
            then
                warn "failed to remove $p"
            fi
        else
            warn "supposedly installed file $p does not exist!"
        fi
    done < "${INSTALLED_MANIFEST}"

    # Remove 'rustlib' directory
    rm -r "${CFG_LIBDIR}/rustlib"
    if [ $? -ne 0 ]
    then
        warn "failed to remove rustlib"
    fi
else
    # There's no manifest. If we were asked to uninstall, then that's a problem.
    if [ -n "${CFG_UNINSTALL}" ]
    then
        err "unable to find installation manifest at ${CFG_LIBDIR}/rustlib"
    fi
fi

# If we're only uninstalling then exit
if [ -n "${CFG_UNINSTALL}" ]
then
    echo
    echo "    Rust is uninstalled. Have a nice day."
    echo
    exit 0
fi

# Create the installed manifest, which we will fill in with absolute file paths
mkdir -p "${CFG_LIBDIR}/rustlib"
touch "${INSTALLED_MANIFEST}"

# Now install, iterate through the new manifest and copy files
while read p; do

    # Decide the destination of the file
    FILE_INSTALL_PATH="${CFG_PREFIX}/$p"

    if echo "$p" | grep "^lib/" > /dev/null
    then
        pp=`echo $p | sed 's/^lib\///'`
        FILE_INSTALL_PATH="${CFG_LIBDIR}/$pp"
    fi

    if echo "$p" | grep "^share/man/" > /dev/null
    then
        pp=`echo $p | sed 's/^share\/man\///'`
        FILE_INSTALL_PATH="${CFG_MANDIR}/$pp"
    fi

    # Make sure there's a directory for it
    umask 022 && mkdir -p "$(dirname ${FILE_INSTALL_PATH})"
    need_ok "directory creation failed"

    # Make the path absolute so we can uninstall it later without
    # starting from the installation cwd
    FILE_INSTALL_PATH_DIRNAME="$(dirname ${FILE_INSTALL_PATH})"
    FILE_INSTALL_PATH_BASENAME="$(basename ${FILE_INSTALL_PATH})"
    FILE_INSTALL_ABS_PATH="$(cd ${FILE_INSTALL_PATH_DIRNAME} && pwd)"
    FILE_INSTALL_PATH="${FILE_INSTALL_ABS_PATH}/${FILE_INSTALL_PATH_BASENAME}"

    # Install the file
    msg "${FILE_INSTALL_PATH}"
    if echo "$p" | grep "^bin/" > /dev/null
    then
        install -m755 "${CFG_SRC_DIR}/$p" "${FILE_INSTALL_PATH}"
    else
        install -m644 "${CFG_SRC_DIR}/$p" "${FILE_INSTALL_PATH}"
    fi
    need_ok "file creation failed"

    # Update the manifest
    echo "${FILE_INSTALL_PATH}" >> "${INSTALLED_MANIFEST}"
    need_ok "failed to update manifest"

# The manifest lists all files to install
done < "${CFG_SRC_DIR}/lib/rustlib/manifest.in"

echo
echo "    Rust is ready to roll."
echo


