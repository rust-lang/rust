#!/bin/sh

TARG_DIR=$1
PREFIX=$2

BINDIR=bin
LIBDIR=lib

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
	LIBDIR=bin
	break
	;;
esac

if [ -z $PREFIX ]; then
    echo "No local rust specified."
    exit 1
fi

if [ ! -e ${PREFIX}/bin/rustc ]; then
    echo "No local rust installed at ${PREFIX}"
    exit 1
fi

if [ -z $TARG_DIR ]; then
    echo "No target directory specified."
    exit 1
fi

cp ${PREFIX}/bin/rustc ${TARG_DIR}/stage0/bin/
cp ${PREFIX}/lib/rustc/${TARG_DIR}/${LIBDIR}/* ${TARG_DIR}/stage0/${LIBDIR}/
cp ${PREFIX}/lib/libextra*${LIB_SUF} ${TARG_DIR}/stage0/${LIBDIR}/
cp ${PREFIX}/lib/librust*${LIB_SUF} ${TARG_DIR}/stage0/${LIBDIR}/
cp ${PREFIX}/lib/libstd*${LIB_SUF} ${TARG_DIR}/stage0/${LIBDIR}/
cp ${PREFIX}/lib/libsyntax*${LIB_SUF} ${TARG_DIR}/stage0/${LIBDIR}/
