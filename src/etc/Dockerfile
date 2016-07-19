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

# When we compile compiler-rt we pass it the llvm-config we just installed on
# the system, but unfortunately it doesn't infer correctly where
# LLVMConfig.cmake is so we need to coerce it a bit...
RUN mkdir -p /usr/lib/llvm-3.7/build/share/llvm
RUN ln -s /usr/share/llvm-3.7/cmake /usr/lib/llvm-3.7/build/share/llvm/cmake

RUN mkdir /build
WORKDIR /build
