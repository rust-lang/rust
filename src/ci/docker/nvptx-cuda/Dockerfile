FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    g++ make file curl ca-certificates python git \
    cmake sudo gdb

# FIXME: setup `ptx-linker` CI for automatic binary releases.
RUN curl -sL https://github.com/denzp/rust-ptx-linker/releases/download/v0.9.0-alpha.1/rust-ptx-linker.linux64.tar.gz | \
    tar -xzvC /usr/bin

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV TARGETS=nvptx64-nvidia-cuda

ENV SCRIPT python2.7 /checkout/x.py test --target $TARGETS \
    src/test/run-make
