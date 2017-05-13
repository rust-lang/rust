FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  gcc libc6-dev qemu curl ca-certificates \
  genext2fs
ENV PATH=$PATH:/rust/bin \
    QEMU=2016-09-07/openbsd-6.0-without-pkgs.qcow2
