FROM fedora:37

RUN dnf update -y && dnf install -y \
      bc \
      bzip2 \
      ca-certificates \
      cmake \
      cpio \
      curl \
      file \
      gcc \
      g++ \
      git \
      make \
      ninja-build \
      python3 \
      qemu \
      edk2-ovmf \
      mtools \
      xz \
      libstdc++-static \
      llvm-devel

ENV ARCH=x86_64

# Create rootfs
WORKDIR /tmp
RUN mkdir rootfs

# Copy over startup script
RUN echo "@echo -off\n" \
         "fs0:\n" \
         "testd --sequential" > rootfs/startup.nsh

# This is here in case we want to use some other version/source for OVMF
RUN mkdir OVMF
RUN cp /usr/share/OVMF/* OVMF

# TODO: What is this?!
# Source of the file: https://github.com/vfdev-5/qemu-rpi2-vexpress/raw/master/vexpress-v2p-ca15-tc1.dtb
RUN curl -O https://ci-mirrors.rust-lang.org/rustc/vexpress-v2p-ca15-tc1.dtb

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_TEST_THREADS=1
ENV DEPLOY=1
ENV RUST_CONFIGURE_ARGS="${RUST_CONFIGURE_ARGS} --qemu-x86_64-uefi-rootfs=/tmp/rootfs --enable-lld --enable-ninja --set target.x86_64-unknown-uefi.linker=rust-lld --set llvm.download-ci-llvm=false"
ENV SCRIPT python3 ../x.py --stage 2 test --host='' --target x86_64-unknown-uefi --warnings warn -v

ENV NO_CHANGE_USER=1
