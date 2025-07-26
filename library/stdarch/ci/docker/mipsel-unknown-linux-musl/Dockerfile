FROM ubuntu:25.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    libc6-dev \
    make \
    qemu-user \
    qemu-system-mips \
    bzip2 \
    curl \
    file

RUN mkdir /toolchain

# Note that this originally came from:
# https://downloads.openwrt.org/snapshots/trunk/malta/generic/OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2
RUN curl -L https://ci-mirrors.rust-lang.org/libc/OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2 | \
      tar xjf - -C /toolchain --strip-components=2

ENV PATH=$PATH:/rust/bin:/toolchain/bin \
    CC_mipsel_unknown_linux_musl=mipsel-openwrt-linux-gcc \
    CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_MUSL_LINKER=mipsel-openwrt-linux-gcc \
    CARGO_TARGET_MIPSEL_UNKNOWN_LINUX_MUSL_RUNNER="qemu-mipsel -L /toolchain"
