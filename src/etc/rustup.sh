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
    echo "rustup: $1"
}

step_msg() {
    msg
    msg "$1"
    msg
}

warn() {
    echo "rustup: WARNING: $1"
}

err() {
    echo "rustup: error: $1"
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
        printf "rustup: %-20s := %.35s ...\n" $1 "$T"
    else
        printf "rustup: %-20s := %s %s\n" $1 "$T" "$2"
    fi
}

probe() {
    local V=$1
    shift
    local P
    local T
    for P
    do
        T=$(which $P 2>&1)
        if [ $? -eq 0 ]
        then
            VER0=$($P --version 2>/dev/null | head -1 \
                |  sed -e 's/[^0-9]*\([vV]\?[0-9.]\+[^ ]*\).*/\1/' )
            if [ $? -eq 0 -a "x${VER0}" != "x" ]
            then
              VER="($VER0)"
            else
              VER=""
            fi
            break
        else
            VER=""
            T=""
        fi
    done
    eval $V=\$T
    putvar $V "$VER"
}

probe_need() {
    local V=$1
    probe $*
    eval VV=\$$V
    if [ -z "$VV" ]
    then
        err "needed, but unable to find any of: $*"
    fi
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

probe_need CFG_CURL  curl

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
valopt prefix "" "set installation prefix"
opt cargo 1 "install cargo with rust"

if [ $HELP -eq 1 ]
then
    echo
    exit 0
fi

step_msg "validating $CFG_SELF args"
validate_opt


# Platform detection copied from `configure`

CFG_OSTYPE=$(uname -s)
CFG_CPUTYPE=$(uname -m)

if [ $CFG_OSTYPE = Darwin -a $CFG_CPUTYPE = i386 ]
then
    # Darwin's `uname -s` lies and always returns i386. We have to use sysctl
    # instead.
    if sysctl hw.optional.x86_64 | grep -q ': 1'
    then
        CFG_CPUTYPE=x86_64
    fi
fi

# The goal here is to come up with the same triple as LLVM would,
# at least for the subset of platforms we're willing to target.

case $CFG_OSTYPE in

    Linux)
        CFG_OSTYPE=unknown-linux-gnu
        ;;

    FreeBSD)
        CFG_OSTYPE=unknown-freebsd
        ;;

    Darwin)
        CFG_OSTYPE=apple-darwin
        ;;

    MINGW32*)
        CFG_OSTYPE=pc-mingw32
        ;;
# Thad's Cygwin identifers below

#   Vista 32 bit
    CYGWIN_NT-6.0)
        CFG_OSTYPE=pc-mingw32
        CFG_CPUTYPE=i686
        ;;

#   Vista 64 bit
    CYGWIN_NT-6.0-WOW64)
        CFG_OSTYPE=w64-mingw32
        CFG_CPUTYPE=x86_64
        ;;

#   Win 7 32 bit
    CYGWIN_NT-6.1)
        CFG_OSTYPE=pc-mingw32
        CFG_CPUTYPE=i686
        ;;

#   Win 7 64 bit
    CYGWIN_NT-6.1-WOW64)
        CFG_OSTYPE=w64-mingw32
        CFG_CPUTYPE=x86_64
        ;;

# We do not detect other OS such as XP/2003 using 64 bit using uname.
# If we want to in the future, we will need to use Cygwin - Chuck's csih helper in /usr/lib/csih/winProductName.exe or alternative.
    *)
        err "unknown OS type: $CFG_OSTYPE"
        ;;
esac


case $CFG_CPUTYPE in

    i386 | i486 | i686 | i786 | x86)
        CFG_CPUTYPE=i686
        ;;

    xscale | arm)
        CFG_CPUTYPE=arm
        ;;

    x86_64 | x86-64 | x64 | amd64)
        CFG_CPUTYPE=x86_64
        ;;

    *)
        err "unknown CPU type: $CFG_CPUTYPE"
esac

# Detect 64 bit linux systems with 32 bit userland and force 32 bit compilation
if [ $CFG_OSTYPE = unknown-linux-gnu -a $CFG_CPUTYPE = x86_64 ]
then
    file -L "$SHELL" | grep -q "x86[_-]64"
    if [ $? != 0 ]; then
        CFG_CPUTYPE=i686
    fi
fi

HOST_TRIPLE="${CFG_CPUTYPE}-${CFG_OSTYPE}"

# Is this a triple we have nightlies for?
case $HOST_TRIPLE in

	x86_64-unknown-linux-gnu)
		;;

	i686-unknown-linux-gnu)
		;;

	x86_64-apple-darwin)
		;;

	i686-apple-darwin)
		;;

	*)
		err "rustup.sh doesn't work for host $HOST_TRIPLE"

esac

msg "host triple: ${HOST_TRIPLE}"

PACKAGE_NAME=rust-nightly
PACKAGE_NAME_AND_TRIPLE="${PACKAGE_NAME}-${HOST_TRIPLE}"
TARBALL_NAME="${PACKAGE_NAME_AND_TRIPLE}.tar.gz"
REMOTE_TARBALL="https://static.rust-lang.org/dist/${TARBALL_NAME}"
TMP_DIR="./rustup-tmp-install"
LOCAL_TARBALL="${TMP_DIR}/${TARBALL_NAME}"
LOCAL_INSTALL_DIR="${TMP_DIR}/${PACKAGE_NAME_AND_TRIPLE}"
LOCAL_INSTALL_SCRIPT="${LOCAL_INSTALL_DIR}/install.sh"

CARGO_PACKAGE_NAME=cargo-nightly
CARGO_PACKAGE_NAME_AND_TRIPLE="${CARGO_PACKAGE_NAME}-${HOST_TRIPLE}"
CARGO_TARBALL_NAME="${CARGO_PACKAGE_NAME_AND_TRIPLE}.tar.gz"
CARGO_REMOTE_TARBALL="https://static.rust-lang.org/cargo-dist/${CARGO_TARBALL_NAME}"
CARGO_LOCAL_TARBALL="${TMP_DIR}/${CARGO_TARBALL_NAME}"
CARGO_LOCAL_INSTALL_DIR="${TMP_DIR}/${CARGO_PACKAGE_NAME_AND_TRIPLE}"
CARGO_LOCAL_INSTALL_SCRIPT="${CARGO_LOCAL_INSTALL_DIR}/install.sh"

rm -Rf "${TMP_DIR}"
need_ok "failed to remove temporary installation directory"

mkdir -p "${TMP_DIR}"
need_ok "failed to create create temporary installation directory"

msg "downloading rust installer"
"${CFG_CURL}" "${REMOTE_TARBALL}" > "${LOCAL_TARBALL}"
if [ $? -ne 0 ]
then
	rm -Rf "${TMP_DIR}"
	err "failed to download installer"
fi

if [ -z "${CFG_DISABLE_CARGO}" ]; then
    msg "downloading cargo installer"
    "${CFG_CURL}" "${CARGO_REMOTE_TARBALL}" > "${CARGO_LOCAL_TARBALL}"
    if [ $? -ne 0 ]
    then
            rm -Rf "${TMP_DIR}"
            err "failed to download cargo installer"
    fi
fi


(cd "${TMP_DIR}" && tar xzf "${TARBALL_NAME}")
if [ $? -ne 0 ]
then
	rm -Rf "${TMP_DIR}"
	err "failed to unpack installer"
fi

MAYBE_UNINSTALL=
if [ -n "${CFG_UNINSTALL}" ]
then
	MAYBE_UNINSTALL="--uninstall"
fi

MAYBE_PREFIX=
if [ -n "${CFG_PREFIX}" ]
then
	MAYBE_PREFIX="--prefix=${CFG_PREFIX}"
fi

sh "${LOCAL_INSTALL_SCRIPT}" "${MAYBE_UNINSTALL}" "${MAYBE_PREFIX}"
if [ $? -ne 0 ]
then
	rm -Rf "${TMP_DIR}"
	err "failed to install Rust"
fi

if [ -z "${CFG_DISABLE_CARGO}" ]; then
    (cd "${TMP_DIR}" && tar xzf "${CARGO_TARBALL_NAME}")
    if [ $? -ne 0 ]
    then
            rm -Rf "${TMP_DIR}"
            err "failed to unpack cargo installer"
    fi

    sh "${CARGO_LOCAL_INSTALL_SCRIPT}" "${MAYBE_UNINSTALL}" "${MAYBE_PREFIX}"
    if [ $? -ne 0 ]
    then
            rm -Rf "${TMP_DIR}"
            err "failed to install Cargo"
    fi
fi

rm -Rf "${TMP_DIR}"
need_ok "couldn't rm temporary installation directory"
