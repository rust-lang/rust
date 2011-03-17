#!/bin/sh

CFG_SRC_DIR=${0%${0##*/}}
CFG_BUILD_DIR=$PWD

echo "configure: recreating config.mk"
echo '' >config.mk

echo "configure: making directories"
for i in \
    boot/fe boot/me boot/be boot/driver boot/util \
    rt/isaac rt/bigint rt/sync rt/test \
    stage0 stage1 stage2 \
    test/run-pass test/compile-pass test/compile-fail
do
    mkdir -p -v $i
done

echo "configure: copying Makefile"
cp -v ${CFG_SRC_DIR}Makefile.in ./Makefile

putvar() {
    local T
    eval T=\$$1
    printf "%-20s := %s\n" $1 $T
    printf "%-20s := %s\n" $1 $T >>config.mk
}

probe() {
    local V=$1
    local P=$2
    local T
    T=$(which $P 2>&1)
    if [ $? -ne 0 ]
    then
        T=""
    fi
    eval $V=\$T
    putvar $V
}

echo "configure: inspecting environment"

CFG_OSTYPE=$(uname -s)
CFG_CPUTYPE=$(uname -m)

putvar CFG_SRC_DIR
putvar CFG_BUILD_DIR
putvar CFG_OSTYPE
putvar CFG_CPUTYPE

echo "configure: looking for programs"
probe CFG_VALGRIND         valgrind
probe CFG_OCAMLC           ocamlc
probe CFG_OCAMLC_OPT       ocamlc.opt
probe CFG_OCAMLOPT         ocamlopt
probe CFG_OCAMLOPT_OPT     ocamlopt.opt
probe CFG_FLEXLINK         flexlink
probe CFG_LLVM_CONFIG      llvm-config

echo "configure: complete"
