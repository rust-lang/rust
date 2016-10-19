#!/bin/sh
# Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

TARG_DIR=$1
PREFIX=$2
RUSTLIBDIR=$3

LIB_DIR=lib
LIB_PREFIX=lib

OS=`uname -s`
case $OS in
    ("Linux"|"FreeBSD"|"DragonFly"|"Bitrig"|"OpenBSD"|"SunOS"|"Haiku")
    BIN_SUF=
    LIB_SUF=.so
    ;;
    ("Darwin")
    BIN_SUF=
    LIB_SUF=.dylib
    ;;
    (*)
    BIN_SUF=.exe
    LIB_SUF=.dll
    LIB_DIR=bin
    LIB_PREFIX=
    ;;
esac

if [ -z $PREFIX ]; then
    echo "No local rust specified."
    exit 1
fi

if [ ! -e ${PREFIX}/bin/rustc${BIN_SUF} ]; then
    echo "No local rust installed at ${PREFIX}"
    exit 1
fi

if [ -z $TARG_DIR ]; then
    echo "No target directory specified."
    exit 1
fi

case "$TARG_DIR" in
--print-rustc-release)
  # not actually copying to TARG_DIR, just print the local rustc version and exit
  ${PREFIX}/bin/rustc${BIN_SUF} --version --verbose | sed -ne 's/^release: //p'
;;
*)

cp ${PREFIX}/bin/rustc${BIN_SUF} ${TARG_DIR}/stage0/bin/
cp ${PREFIX}/${LIB_DIR}/${RUSTLIBDIR}/${TARG_DIR}/${LIB_DIR}/* ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}arena*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}extra*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}rust*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}std*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}syntax*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}flate*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}fmt_macros*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}getopts*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}graphviz*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}log*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}rbml*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}serialize*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}term*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}proc_macro*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/

# do not fail if one of the above fails, as all we need is a working rustc!
exit 0

esac
