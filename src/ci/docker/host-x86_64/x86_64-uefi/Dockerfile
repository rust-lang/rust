FROM ubuntu:22.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      bc \
      bzip2 \
      ca-certificates \
      cmake \
      cpio \
      curl \
      file \
      g++ \
      git \
      libc6-dev \
      make \
      ninja-build \
      python3 \
      qemu-system \
      xz-utils

ENV ARCH=x86_64

# Create rootfs
WORKDIR /tmp
RUN mkdir rootfs

# Copy over startup script
COPY scripts/qemu_uefi.nsh rootfs/startup.nsh

# OVMF Stuff
RUN mkdir OVMF
COPY OVMF/OVMF_CODE.fd OVMF/OVMF_CODE.fd
COPY OVMF/OVMF_VARS.fd OVMF/OVMF_VARS.fd
COPY OVMF/UefiShell.iso OVMF/UefiShell.iso

# Config
COPY host-x86_64/x86_64-uefi/config.toml config.toml

# TODO: What is this?!
# Source of the file: https://github.com/vfdev-5/qemu-rpi2-vexpress/raw/master/vexpress-v2p-ca15-tc1.dtb
RUN curl -O https://ci-mirrors.rust-lang.org/rustc/vexpress-v2p-ca15-tc1.dtb

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS="${RUST_CONFIGURE_ARGS} --qemu-uefi-rootfs=/tmp/rootfs --enable-lld"
ENV SCRIPT python3 ../x.py --stage 2 test --host='' --target x86_64-unknown-uefi --warnings warn --config /tmp/config.toml -v

ENV NO_CHANGE_USER=1
