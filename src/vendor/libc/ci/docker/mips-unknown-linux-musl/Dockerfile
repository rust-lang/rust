FROM ubuntu:16.10

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        gcc libc6-dev qemu-user ca-certificates qemu-system-mips curl \
        bzip2

RUN mkdir /toolchain
RUN curl -L https://downloads.openwrt.org/snapshots/trunk/ar71xx/generic/OpenWrt-SDK-ar71xx-generic_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2 | \
      tar xjf - -C /toolchain --strip-components=1

ENV PATH=$PATH:/rust/bin:/toolchain/staging_dir/toolchain-mips_34kc_gcc-5.3.0_musl-1.1.15/bin \
    CC_mips_unknown_linux_musl=mips-openwrt-linux-gcc \
    CARGO_TARGET_MIPS_UNKNOWN_LINUX_MUSL_LINKER=mips-openwrt-linux-gcc
