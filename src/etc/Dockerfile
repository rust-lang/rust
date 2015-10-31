FROM ubuntu:latest

RUN echo 'deb http://llvm.org/apt/trusty/ llvm-toolchain-trusty-3.7 main' | \
        sudo tee -a /etc/apt/sources.list
RUN echo 'deb-src http://llvm.org/apt/trusty/ llvm-toolchain-trusty-3.7 main' | \
        sudo tee -a /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install curl make g++ python2.7 git zlib1g-dev libedit-dev
RUN apt-get -y --force-yes install llvm-3.7-tools

RUN mkdir /build
WORKDIR /build
