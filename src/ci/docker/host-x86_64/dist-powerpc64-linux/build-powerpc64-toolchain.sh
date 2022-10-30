#!/usr/bin/env bash
set -ex

source shared.sh

mkdir build
cd build
cp ../powerpc64-linux-gnu.config .config
hide_output ct-ng build
cd ..
rm -rf build
