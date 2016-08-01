FROM ubuntu:xenial

# curl
#   Download stage0, see src/bootstrap/bootstrap.py
# g++
#   Compile LLVM binding in src/rustllvm
# gdb
#   Used to run tests in src/test/debuginfo
# git
#   Get commit hash and commit date in version string
# make
#   Run build scripts in mk
# libedit-dev zlib1g-dev
#   LLVM dependencies as packaged in Ubuntu
#   (They are optional, but Ubuntu package enables them)
# llvm-3.7-dev (installed by llvm-3.7-tools)
#   LLVM
# llvm-3.7-tools
#   FileCheck is used to run tests in src/test/codegen

RUN apt-get update && apt-get -y install \
    curl g++ gdb git make \
    libedit-dev zlib1g-dev \
    llvm-3.7-tools cmake

RUN mkdir /build
WORKDIR /build
