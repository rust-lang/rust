FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
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

COPY host-x86_64/dist-various-1/install-aarch64-none-elf.sh /build
RUN ./install-aarch64-none-elf.sh

COPY host-x86_64/dist-various-1/install-riscv64-none-elf.sh /build
RUN ./install-riscv64-none-elf.sh

COPY host-x86_64/dist-various-1/install-riscv32-none-elf.sh /build
RUN ./install-riscv32-none-elf.sh

COPY host-x86_64/dist-various-1/install-emscripten.sh /build
RUN ./install-emscripten.sh

# Add Emscripten to PATH
ENV PATH="/build/emsdk:/build/emsdk/upstream/emscripten:/build/emsdk/node/current/bin:${PATH}"

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
    CC=arm-linux-gnueabihf-gcc CFLAGS="-march=armv7-a+fp" \
    CXX=arm-linux-gnueabihf-g++ CXXFLAGS="-march=armv7-a+fp" \
    bash musl.sh armv7hf

ENV RUN_MAKE_TARGETS=thumbv6m-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7m-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7em-none-eabi
ENV RUN_MAKE_TARGETS=$RUN_MAKE_TARGETS,thumbv7em-none-eabihf

ENV TARGETS=wasm32-unknown-emscripten
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
ENV TARGETS=$TARGETS,riscv32im-unknown-none-elf
ENV TARGETS=$TARGETS,riscv32imc-unknown-none-elf
ENV TARGETS=$TARGETS,riscv32imac-unknown-none-elf
ENV TARGETS=$TARGETS,riscv32imafc-unknown-none-elf
ENV TARGETS=$TARGETS,riscv64imac-unknown-none-elf
ENV TARGETS=$TARGETS,riscv64gc-unknown-none-elf
ENV TARGETS=$TARGETS,armebv7r-none-eabi
ENV TARGETS=$TARGETS,armebv7r-none-eabihf
ENV TARGETS=$TARGETS,armv7r-none-eabi
ENV TARGETS=$TARGETS,armv7r-none-eabihf
ENV TARGETS=$TARGETS,thumbv7neon-unknown-linux-gnueabihf
ENV TARGETS=$TARGETS,armv7a-none-eabi

ENV CFLAGS_armv5te_unknown_linux_musleabi="-march=armv5te -marm -mfloat-abi=soft" \
    CFLAGS_arm_unknown_linux_musleabi="-march=armv6 -marm" \
    CFLAGS_arm_unknown_linux_musleabihf="-march=armv6 -marm -mfpu=vfp" \
    CFLAGS_armv7_unknown_linux_musleabihf="-march=armv7-a+fp" \
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
    CC_riscv32i_unknown_none_elf=riscv32-unknown-elf-gcc \
    CFLAGS_riscv32i_unknown_none_elf=-march=rv32i -mabi=ilp32 \
    CC_riscv32im_unknown_none_elf=riscv32-unknown-elf-gcc \
    CFLAGS_riscv32im_unknown_none_elf=-march=rv32im -mabi=ilp32 \
    CC_riscv32imc_unknown_none_elf=riscv32-unknown-elf-gcc \
    CFLAGS_riscv32imc_unknown_none_elf=-march=rv32imc -mabi=ilp32 \
    CC_riscv32imac_unknown_none_elf=riscv32-unknown-elf-gcc \
    CFLAGS_riscv32imac_unknown_none_elf=-march=rv32imac -mabi=ilp32 \
    CC_riscv32imafc_unknown_none_elf=riscv32-unknown-elf-gcc \
    CFLAGS_riscv32imafc_unknown_none_elf=-march=rv32imafc -mabi=ilp32 \
    CC_riscv64imac_unknown_none_elf=riscv64-unknown-elf-gcc \
    CFLAGS_riscv64imac_unknown_none_elf=-march=rv64imac -mabi=lp64 \
    CC_riscv64gc_unknown_none_elf=riscv64-unknown-elf-gcc \
    CFLAGS_riscv64gc_unknown_none_elf=-march=rv64gc -mabi=lp64

ENV RUST_CONFIGURE_ARGS \
      --musl-root-armv5te=/musl-armv5te \
      --musl-root-arm=/musl-arm \
      --musl-root-armhf=/musl-armhf \
      --musl-root-armv7hf=/musl-armv7hf \
      --disable-docs

ENV SCRIPT \
      python3 ../x.py --stage 2 test --host='' --target $RUN_MAKE_TARGETS tests/run-make tests/run-make-cargo && \
      python3 ../x.py dist --host='' --target $TARGETS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
