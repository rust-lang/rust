# based on armhf-gnu/Dockerfile
FROM ubuntu:20.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    bc \
    bison \
    ca-certificates \
    cmake \
    cpio \
    curl \
    debian-ports-archive-keyring \
    debootstrap \
    flex \
    gcc \
    gcc-riscv64-linux-gnu \
    git \
    g++-riscv64-linux-gnu \
    g++ \
    libc6-dev \
    libc6-dev-riscv64-cross \
    make \
    ninja-build \
    patch \
    python3 \
    qemu-system-misc \
    xz-utils

ENV ARCH=riscv
ENV CROSS_COMPILE=riscv64-linux-gnu-

WORKDIR /build

# From https://github.com/michaeljclark/busybear-linux/blob/master/conf/linux.config
COPY host-x86_64/riscv64gc-linux/linux.config /build

# Compile the kernel that we're going to be emulating with. This is
# basically just done to be compatible with the QEMU target that we're going
# to be using when running tests.
RUN curl https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.6.16.tar.xz | tar xJf - && \
    cp linux.config linux-5.6.16/.config && \
    cd /build/linux-5.6.16 && \
    make olddefconfig && \
    make -j$(nproc) vmlinux && \
    cp vmlinux /tmp && \
    rm -rf linux-5.6.16

# Compile an instance of busybox as this provides a lightweight system and init
# binary which we will boot into. Only trick here is configuring busybox to
# build static binaries.
RUN curl https://busybox.net/downloads/busybox-1.31.1.tar.bz2 | tar xjf -
COPY host-x86_64/riscv64gc-linux/0001-Remove-stime-function-calls.patch /build/busybox-1.31.1/
RUN cd /build/busybox-1.31.1 && \
    patch -p1 -i 0001-Remove-stime-function-calls.patch && \
    make defconfig && \
    sed -i 's/.*CONFIG_STATIC.*/CONFIG_STATIC=y/' .config && \
    make -j$(nproc) && \
    make install && \
    mv _install /tmp/rootfs && \
    cd /build && \
    rm -rf busybox-1.31.1

# Download the ubuntu rootfs, which we'll use as a chroot for all our tests
# This is only needed to provide /lib/* and /usr/lib/*
WORKDIR /tmp
RUN debootstrap --variant=minbase --arch=riscv64 --foreign focal /tmp/rootfs/ubuntu
RUN cd rootfs && mkdir proc sys dev etc etc/init.d
# rootfs/ubuntu/proc is in a weird state (access fails with ELOOP) until
# rootfs/ubuntu/debootstrap/debootstrap --second-stage is run (under emulation),
# but this takes ages. Instead hack it into a good enough state.
# /proc is used by std::env::current_exe() (which is roughly
# `readlink /proc/self/exe`)
RUN cd rootfs/ubuntu && rm -rf proc && mkdir proc

# Copy over our init script, which starts up our test server and also a few other
# misc tasks
COPY scripts/qemu-bare-bones-rcS rootfs/etc/init.d/rcS
RUN chmod +x rootfs/etc/init.d/rcS

# Helper to quickly fill the entropy pool in the kernel
COPY scripts/qemu-bare-bones-addentropy.c /tmp/addentropy.c
RUN riscv64-linux-gnu-gcc addentropy.c -o rootfs/addentropy -static

# download and build the riscv bootloader
RUN git clone https://github.com/riscv/riscv-pk
WORKDIR /tmp/riscv-pk
# This revision fixes a fault in recent QEMU from 64-bit accesses to the PLIC
# commits later than this one should work too
RUN git checkout 7d8b7c0dab72108e3ea7bb7744d3f6cc907c7ef4
RUN mkdir build && cd build && \
    ../configure --with-payload=/tmp/vmlinux --host=riscv64-linux-gnu && \
    make -j$(nproc) && \
    cp bbl /tmp
WORKDIR /tmp
RUN rm -rf /tmp/riscv-pk

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS --qemu-riscv64-rootfs=/tmp/rootfs
ENV SCRIPT python3 ../x.py --stage 2 test --target riscv64gc-unknown-linux-gnu

ENV NO_CHANGE_USER=1
