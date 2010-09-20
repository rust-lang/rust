#!/bin/sh
make -k $1.x86
DYLD_LIBRARY_PATH=../../.. ./$1.x86
