#!/bin/sh

CFG_SRC_DIR=${0%${0##*/}}
CFG_BUILD_DIR=$PWD

CFG_OSTYPE=$(uname -s)
CFG_CPUTYPE=$(uname -m)

echo "configuring on $CFG_CPUTYPE $CFG_OSTYPE"

echo "setting up build directories"
for i in boot/{fe,me,be,driver,util} \
         rt/{isaac,bigint,sync,test} \
         stage{0,1,2}                \
         test/{run-pass,compile-{pass,fail}}
do
    mkdir -p -v $i
done

CFG_VALGRIND=$(sh which valgrind)
CFG_OCAMLC_OPT=$(sh which ocamlc.opt)

echo "copying Makefile"
cp -v ${CFG_SRC_DIR}Makefile.in ./Makefile

echo "writing config.mk"
cat >config.mk <<EOF

CFG_OSTYPE        := $CFG_OSTYPE
CFG_CPUTYPE       := $CFG_CPUTYPE
CFG_SRC_DIR       := $CFG_SRC_DIR
CFG_BUILD_DIR     := $CFG_BUILD_DIR

CFG_VALGRIND      := $CFG_VALGRIND
CFG_OCAMLC_OPT    := $CFG_OCAMLC_OPT

EOF

echo "configured ok"
