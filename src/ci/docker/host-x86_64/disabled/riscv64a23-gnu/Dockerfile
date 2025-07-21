FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    cmake \
    cpio \
    curl \
    file \
    flex \
    bison \
    g++ \
    g++-riscv64-linux-gnu \
    git \
    libc6-dev \
    libc6-dev-riscv64-cross \
    libssl-dev \
    make \
    ninja-build \
    python3 \
    xz-utils \
    opensbi \
    u-boot-qemu \
    libslirp0 \
    build-essential \
    pkg-config \
    libglib2.0-dev \
    libpixman-1-dev \
    libsdl2-dev \
    libfdt-dev \
    python3 \
    python3-pip

ENV ARCH=riscv \
    CROSS_COMPILE=riscv64-linux-gnu-

WORKDIR /build

# From https://github.com/michaeljclark/busybear-linux/blob/master/conf/linux.config
COPY host-x86_64/riscv64a23-gnu/linux.config /build

# qemu v10.0.2 fully support
RUN curl https://gitlab.com/qemu-project/qemu/-/archive/v10.0.2/qemu-v10.0.2.tar.bz2 | tar xjf - && \
    cd qemu-v10.0.2 && \
    ./configure --target-list=riscv64-softmmu \
    --enable-sdl --enable-debug --enable-fdt --enable-slirp && \
    make -j && make install

# use the opensbi fw from apt-get install
RUN cp /usr/lib/riscv64-linux-gnu/opensbi/generic/fw_jump.bin /tmp

# Compile the kernel that we're going to be emulating with. This is
# basically just done to be compatible with the QEMU target that we're going
# to be using when running tests.
RUN curl https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.97.tar.xz | tar xJf - && \
    cp linux.config linux-6.6.97/.config && \
    cd /build/linux-6.6.97 && \
    make olddefconfig && \
    make -j$(nproc) Image && \
    cp arch/riscv/boot/Image /tmp && \
    rm -rf linux-6.6.97

# Compile an instance of busybox as this provides a lightweight system and init
# binary which we will boot into. Only trick here is configuring busybox to
# build static binaries.
RUN curl https://www.busybox.net/downloads/busybox-1.37.0.tar.bz2 | tar xjf - && \
    cd busybox-1.37.0 && \
    make defconfig && \
    sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config && \
    sed -i 's/^CONFIG_TC=y$/# CONFIG_TC is not set/' .config && \
    sed -i 's/CONFIG_SHA1_HWACCEL=y/# CONFIG_SHA1_HWACCEL is not set/' .config && \
    sed -i 's/CONFIG_SHA256_HWACCEL=y/# CONFIG_SHA256_HWACCEL is not set/' .config && \
    make -j$(nproc) && \
    make install && \
    mv _install /tmp/rootfs && \
    cd /build && \
    rm -rf busybox-1.37.0

# Download the ubuntu rootfs, which we'll use as a chroot for all our tests.
WORKDIR /tmp
RUN mkdir rootfs/ubuntu
RUN curl https://cdimage.ubuntu.com/ubuntu-base/releases/24.04/release/ubuntu-base-24.04.2-base-riscv64.tar.gz | \
      tar xzf - -C rootfs/ubuntu && \
      cd rootfs && mkdir proc sys dev etc etc/init.d

# Copy over our init script, which starts up our test server and also a few other
# misc tasks
COPY scripts/qemu-bare-bones-rcS rootfs/etc/init.d/rcS
RUN chmod +x rootfs/etc/init.d/rcS

# Helper to quickly fill the entropy pool in the kernel
COPY scripts/qemu-bare-bones-addentropy.c /tmp/addentropy.c
RUN riscv64-linux-gnu-gcc addentropy.c -o rootfs/addentropy -static

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# Avoid "fatal: detected dubious ownership in repository at '/checkout'" error
RUN git config --global --add safe.directory "*"

ENV RUST_CONFIGURE_ARGS \
    --set target.riscv64a23-unknown-linux-gnu.linker=riscv64-linux-gnu-gcc \
    --set target.riscv64a23-unknown-linux-gnu.qemu-rootfs=/tmp/rootfs
ENV SCRIPT python3 ../x.py --stage 2 test --host='' --target riscv64a23-unknown-linux-gnu 

ENV NO_CHANGE_USER=1
