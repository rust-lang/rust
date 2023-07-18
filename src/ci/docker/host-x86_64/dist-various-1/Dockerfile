FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  g++ \
  automake \
  bison \
  bzip2 \
  flex \
  help2man \
  libtool-bin \
  texinfo \
  unzip \
  wget \
  xz-utils \
  libncurses-dev \
  gawk \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  xz-utils \
  zlib1g-dev \
  g++-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf \
  g++-aarch64-linux-gnu \
  g++-mips64-linux-gnuabi64 \
  g++-mips64el-linux-gnuabi64 \
  gcc-arm-none-eabi \
  gcc-sparc64-linux-gnu \
  libc6-dev-sparc64-cross \
  bzip2 \
  patch \
  libssl-dev \
  pkg-config \
  libnewlib-arm-none-eabi \
  qemu-system-arm \
# software-properties-common for the add-apt-repository command
  software-properties-common

WORKDIR /build

COPY host-x86_64/dist-various-1/install-x86_64-redox.sh /build
RUN ./install-x86_64-redox.sh

COPY host-x86_64/dist-various-1/install-mips-musl.sh /build
RUN ./install-mips-musl.sh

COPY host-x86_64/dist-various-1/install-mipsel-musl.sh /build
RUN ./install-mipsel-musl.sh

COPY host-x86_64/dist-various-1/install-aarch64-none-elf.sh /build
RUN ./install-aarch64-none-elf.sh

# Suppress some warnings in the openwrt toolchains we downloaded
ENV STAGING_DIR=/tmp

COPY scripts/musl.sh /build
RUN env \
    CC=arm-linux-gnueabi-gcc CFLAGS="-march=armv5te -marm -mfloat-abi=soft" \
    CXX=arm-linux-gnueabi-g++ CXXFLAGS="-march=armv5te -marm -mfloat-abi=soft" \
    bash musl.sh armv5te && \
    env \
    CC=arm-linux-gnueabi-gcc CFLAGS="-march=armv6 -marm" \
    CXX=arm-linux-gnueabi-g++ CXXFLAGS="-march=armv6 -marm" \
    bash musl.sh arm && \
    env \
    CC=arm-linux-gnueabihf-gcc CFLAGS="-march=armv6 -marm -mfpu=vfp" \
    CXX=arm-linux-gnueabihf-g++ CXXFLAGS="-march=armv6 -marm -mfpu=vfp" \
    bash musl.sh armhf && \
    env \
    CC=arm-linux-gnueabihf-gcc CFLAGS="-march=armv7-a" \
    CXX=arm-linux-gnueabihf-g++ CXXFLAGS="-march=armv7-a" \
    bash musl.sh armv7hf && \
    env \
    CC=mips-openwrt-linux-gcc \
    CXX=mips-openwrt-linux-g++ \
    bash musl.sh mips && \
    env \
    CC=mipsel-openwrt-linux-gcc \
    CXX=mipsel-openwrt-linux-g++ \
    bash musl.sh mipsel && \
    env \
    CC=mips64-linux-gnuabi64-gcc \
    CXX=mips64-linux-gnuabi64-g++ \
    bash musl.sh mips64 && \
    env \
    CC=mips64el-linux-gnuabi64-gcc \
    CXX=mips64el-linux-gnuabi64-g++ \
    bash musl.sh mips64el && \
    rm -rf /build/*

# FIXME(mozilla/sccache#235) this shouldn't be necessary but is currently
# necessary to disambiguate the mips compiler with the mipsel compiler. We want
# to give these two wrapper scripts (currently identical ones) different hashes
# to ensure that sccache understands that they're different compilers.
RUN \
  echo "# a" >> /usr/local/mips-linux-musl/bin/mips-openwrt-linux-musl-wrapper.sh && \
  echo "# b" >> /usr/local/mipsel-linux-musl/bin/mipsel-openwrt-linux-musl-wrapper.sh

ENV RUN_MAKE_TARGETS=thumbv6m-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7m-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7em-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7em-none-eabihf

ENV TARGETS=asmjs-unknown-emscripten
ENV TARGETS=$TARGETS,wasm32-unknown-emscripten
ENV TARGETS=$TARGETS,mips-unknown-linux-musl
ENV TARGETS=$TARGETS,mipsel-unknown-linux-musl
ENV TARGETS=$TARGETS,mips64-unknown-linux-muslabi64
ENV TARGETS=$TARGETS,mips64el-unknown-linux-muslabi64
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabi
ENV TARGETS=$TARGETS,arm-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,armv5te-unknown-linux-gnueabi
ENV TARGETS=$TARGETS,armv5te-unknown-linux-musleabi
ENV TARGETS=$TARGETS,armv7-unknown-linux-musleabihf
ENV TARGETS=$TARGETS,aarch64-unknown-none
ENV TARGETS=$TARGETS,aarch64-unknown-none-softfloat
ENV TARGETS=$TARGETS,sparc64-unknown-linux-gnu
ENV TARGETS=$TARGETS,x86_64-unknown-redox
ENV TARGETS=$TARGETS,thumbv6m-none-eabi
ENV TARGETS=$TARGETS,thumbv7m-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabihf
ENV TARGETS=$TARGETS,thumbv8m.base-none-eabi
ENV TARGETS=$TARGETS,thumbv8m.main-none-eabi
ENV TARGETS=$TARGETS,thumbv8m.main-none-eabihf
ENV TARGETS=$TARGETS,riscv32i-unknown-none-elf
ENV TARGETS=$TARGETS,riscv32imc-unknown-none-elf
ENV TARGETS=$TARGETS,riscv32imac-unknown-none-elf
ENV TARGETS=$TARGETS,riscv64imac-unknown-none-elf
ENV TARGETS=$TARGETS,riscv64gc-unknown-none-elf
ENV TARGETS=$TARGETS,armebv7r-none-eabi
ENV TARGETS=$TARGETS,armebv7r-none-eabihf
ENV TARGETS=$TARGETS,armv7r-none-eabi
ENV TARGETS=$TARGETS,armv7r-none-eabihf
ENV TARGETS=$TARGETS,thumbv7neon-unknown-linux-gnueabihf
ENV TARGETS=$TARGETS,armv7a-none-eabi

# riscv targets currently do not need a C compiler, as compiler_builtins
# doesn't currently have it enabled, and the riscv gcc compiler is not
# installed.
ENV CFLAGS_armv5te_unknown_linux_musleabi="-march=armv5te -marm -mfloat-abi=soft" \
    CFLAGS_arm_unknown_linux_musleabi="-march=armv6 -marm" \
    CFLAGS_arm_unknown_linux_musleabihf="-march=armv6 -marm -mfpu=vfp" \
    CFLAGS_armv7_unknown_linux_musleabihf="-march=armv7-a" \
    CC_mipsel_unknown_linux_musl=mipsel-openwrt-linux-gcc \
    CC_mips_unknown_linux_musl=mips-openwrt-linux-gcc \
    CC_mips64el_unknown_linux_muslabi64=mips64el-linux-gnuabi64-gcc \
    CC_mips64_unknown_linux_muslabi64=mips64-linux-gnuabi64-gcc \
    CC_sparc64_unknown_linux_gnu=sparc64-linux-gnu-gcc \
    CC_x86_64_unknown_redox=x86_64-unknown-redox-gcc \
    CC_thumbv7neon_unknown_linux_gnueabihf=arm-linux-gnueabihf-gcc \
    AR_thumbv7neon_unknown_linux_gnueabihf=arm-linux-gnueabihf-ar \
    CXX_thumbv7neon_unknown_linux_gnueabihf=arm-linux-gnueabihf-g++ \
    CC_armv7a_none_eabi=arm-none-eabi-gcc \
    CC_armv7a_none_eabihf=arm-none-eabi-gcc \
    CFLAGS_armv7a_none_eabi=-march=armv7-a \
    CFLAGS_armv7a_none_eabihf=-march=armv7-a+vfpv3 \
    CC_aarch64_unknown_none_softfloat=aarch64-none-elf-gcc \
    CFLAGS_aarch64_unknown_none_softfloat=-mstrict-align -march=armv8-a+nofp+nosimd \
    CC_aarch64_unknown_none=aarch64-none-elf-gcc \
    CFLAGS_aarch64_unknown_none=-mstrict-align -march=armv8-a+fp+simd \
    CC_riscv32i_unknown_none_elf=false \
    CC_riscv32imc_unknown_none_elf=false \
    CC_riscv32imac_unknown_none_elf=false \
    CC_riscv64imac_unknown_none_elf=false \
    CC_riscv64gc_unknown_none_elf=false

ENV RUST_CONFIGURE_ARGS \
      --musl-root-armv5te=/musl-armv5te \
      --musl-root-arm=/musl-arm \
      --musl-root-armhf=/musl-armhf \
      --musl-root-armv7hf=/musl-armv7hf \
      --musl-root-mips=/musl-mips \
      --musl-root-mipsel=/musl-mipsel \
      --musl-root-mips64=/musl-mips64 \
      --musl-root-mips64el=/musl-mips64el \
      --disable-docs

ENV SCRIPT \
      python3 ../x.py --stage 2 test --host='' --target $RUN_MAKE_TARGETS tests/run-make && \
      python3 ../x.py dist --host='' --target $TARGETS

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
