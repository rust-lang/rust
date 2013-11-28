#!/bin/sh

TARG_DIR=$1
PREFIX=$2

LIB_DIR=lib
LIB_PREFIX=lib

OS=`uname -s`
case $OS in
    ("Linux"|"FreeBSD")
	BIN_SUF=
	LIB_SUF=.so
	break
	;;
    ("Darwin")
	BIN_SUF=
	LIB_SUF=.dylib
	break
	;;
    (*)
	BIN_SUF=.exe
	LIB_SUF=.dll
	LIB_DIR=bin
	LIB_PREFIX=
	break
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

cp ${PREFIX}/bin/rustc${BIN_SUF} ${TARG_DIR}/stage0/bin/
cp ${PREFIX}/${LIB_DIR}/rustc/${TARG_DIR}/${LIB_DIR}/* ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}extra*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}rust*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}std*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
cp ${PREFIX}/${LIB_DIR}/${LIB_PREFIX}syntax*${LIB_SUF} ${TARG_DIR}/stage0/${LIB_DIR}/
