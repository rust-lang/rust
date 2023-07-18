FROM ubuntu:20.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

# Enable source repositories, which are disabled by default on Ubuntu >= 18.04
RUN sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list

RUN apt-get update && apt-get build-dep -y clang llvm && apt-get install -y --no-install-recommends \
  build-essential \
# gcc-multilib can not be installed together with gcc-arm-linux-gnueabi
  g++-8-multilib \
  libedit-dev \
  libgmp-dev \
  libisl-dev \
  libmpc-dev \
  libmpfr-dev \
  libtinfo5 \
  ninja-build \
  nodejs \
  python3-dev \
  software-properties-common \
  unzip \
  # Needed for apt-key to work:
  dirmngr \
  gpg-agent \
  g++-8-arm-linux-gnueabi

RUN apt-key adv --batch --yes --keyserver keyserver.ubuntu.com --recv-keys 74DA7924C5513486
RUN add-apt-repository -y 'deb https://apt.dilos.org/dilos dilos2 main'

ENV \
    AR_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang \
    CFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang++ \
    CXXFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_x86_64_unknown_fuchsia="--target=x86_64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib" \
    AR_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-ar \
    CC_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang \
    CFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    CXX_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang++ \
    CXXFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot -I/usr/local/core-linux-amd64-fuchsia-sdk/pkg/fdio/include" \
    LDFLAGS_aarch64_unknown_fuchsia="--target=aarch64-unknown-fuchsia --sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot -L/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/lib" \
    AR_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-ar \
    CC_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-gcc \
    CXX_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-g++ \
    AR_x86_64_pc_solaris=x86_64-pc-solaris2.10-ar \
    CC_x86_64_pc_solaris=x86_64-pc-solaris2.10-gcc \
    CXX_x86_64_pc_solaris=x86_64-pc-solaris2.10-g++ \
    AR_x86_64_sun_solaris=x86_64-sun-solaris2.10-ar \
    CC_x86_64_sun_solaris=x86_64-sun-solaris2.10-gcc \
    CXX_x86_64_sun_solaris=x86_64-sun-solaris2.10-g++ \
    CC_armv7_unknown_linux_gnueabi=arm-linux-gnueabi-gcc-8 \
    CXX_armv7_unknown_linux_gnueabi=arm-linux-gnueabi-g++-8 \
    AR_x86_64_fortanix_unknown_sgx=ar \
    CC_x86_64_fortanix_unknown_sgx=clang-11 \
    CFLAGS_x86_64_fortanix_unknown_sgx="-D__ELF__ -isystem/usr/include/x86_64-linux-gnu -mlvi-hardening -mllvm -x86-experimental-lvi-inline-asm-hardening" \
    CXX_x86_64_fortanix_unknown_sgx=clang++-11 \
    CXXFLAGS_x86_64_fortanix_unknown_sgx="-D__ELF__ -isystem/usr/include/x86_64-linux-gnu -mlvi-hardening -mllvm -x86-experimental-lvi-inline-asm-hardening" \
    AR_i686_unknown_freebsd=i686-unknown-freebsd12-ar \
    CC_i686_unknown_freebsd=i686-unknown-freebsd12-clang \
    CXX_i686_unknown_freebsd=i686-unknown-freebsd12-clang++ \
    CC_aarch64_unknown_uefi=clang-11 \
    CXX_aarch64_unknown_uefi=clang++-11 \
    CC_i686_unknown_uefi=clang-11 \
    CXX_i686_unknown_uefi=clang++-11 \
    CC_x86_64_unknown_uefi=clang-11 \
    CXX_x86_64_unknown_uefi=clang++-11 \
    CC=gcc-8 \
    CXX=g++-8

WORKDIR /build
COPY scripts/musl.sh /build
RUN env \
    CC=arm-linux-gnueabi-gcc-8 CFLAGS="-march=armv7-a" \
    CXX=arm-linux-gnueabi-g++-8 CXXFLAGS="-march=armv7-a" \
    bash musl.sh armv7 && \
    rm -rf /build/*

WORKDIR /tmp
COPY host-x86_64/dist-various-2/shared.sh /tmp/
COPY host-x86_64/dist-various-2/build-fuchsia-toolchain.sh /tmp/
RUN /tmp/build-fuchsia-toolchain.sh
COPY host-x86_64/dist-various-2/build-solaris-toolchain.sh /tmp/
RUN /tmp/build-solaris-toolchain.sh x86_64  amd64   solaris-i386  pc
# Build deprecated target 'x86_64-sun-solaris2.10' until removed
RUN /tmp/build-solaris-toolchain.sh x86_64  amd64   solaris-i386  sun
RUN /tmp/build-solaris-toolchain.sh sparcv9 sparcv9 solaris-sparc sun
COPY host-x86_64/dist-various-2/build-x86_64-fortanix-unknown-sgx-toolchain.sh /tmp/
RUN /tmp/build-x86_64-fortanix-unknown-sgx-toolchain.sh

COPY host-x86_64/dist-various-2/build-wasi-toolchain.sh /tmp/
RUN /tmp/build-wasi-toolchain.sh

COPY scripts/freebsd-toolchain.sh /tmp/
RUN /tmp/freebsd-toolchain.sh i686

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV CARGO_TARGET_X86_64_UNKNOWN_FUCHSIA_AR /usr/local/bin/llvm-ar
ENV CARGO_TARGET_X86_64_UNKNOWN_FUCHSIA_RUSTFLAGS \
-C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot \
-Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/sysroot/lib \
-Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/x64/lib
ENV CARGO_TARGET_AARCH64_UNKNOWN_FUCHSIA_AR /usr/local/bin/llvm-ar
ENV CARGO_TARGET_AARCH64_UNKNOWN_FUCHSIA_RUSTFLAGS \
-C link-arg=--sysroot=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot \
-Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/sysroot/lib \
-Lnative=/usr/local/core-linux-amd64-fuchsia-sdk/arch/arm64/lib

ENV TARGETS=x86_64-unknown-fuchsia
ENV TARGETS=$TARGETS,aarch64-unknown-fuchsia
ENV TARGETS=$TARGETS,wasm32-unknown-unknown
ENV TARGETS=$TARGETS,wasm32-wasi
ENV TARGETS=$TARGETS,sparcv9-sun-solaris
ENV TARGETS=$TARGETS,x86_64-pc-solaris
ENV TARGETS=$TARGETS,x86_64-sun-solaris
ENV TARGETS=$TARGETS,x86_64-unknown-linux-gnux32
ENV TARGETS=$TARGETS,x86_64-fortanix-unknown-sgx
ENV TARGETS=$TARGETS,nvptx64-nvidia-cuda
ENV TARGETS=$TARGETS,armv7-unknown-linux-gnueabi
ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabi
ENV TARGETS=$TARGETS,i686-unknown-freebsd
ENV TARGETS=$TARGETS,x86_64-unknown-none
ENV TARGETS=$TARGETS,aarch64-unknown-uefi
ENV TARGETS=$TARGETS,i686-unknown-uefi
ENV TARGETS=$TARGETS,x86_64-unknown-uefi

# As per https://bugs.launchpad.net/ubuntu/+source/gcc-defaults/+bug/1300211
# we need asm in the search path for gcc-8 (for gnux32) but not in the search path of the
# cross compilers.
# Luckily one of the folders is /usr/local/include so symlink /usr/include/asm-generic there
RUN ln -s /usr/include/asm-generic /usr/local/include/asm

ENV RUST_CONFIGURE_ARGS --enable-extended --enable-lld --disable-docs \
  --set target.wasm32-wasi.wasi-root=/wasm32-wasi \
  --musl-root-armv7=/musl-armv7

ENV SCRIPT python3 ../x.py dist --host='' --target $TARGETS
