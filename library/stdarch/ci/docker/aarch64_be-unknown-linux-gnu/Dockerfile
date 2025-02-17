FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  ca-certificates \
  libc6-dev \
  libc6-dev-arm64-cross \
  qemu-user \
  make \
  file \
  clang-19 \
  curl \
  xz-utils \
  lld

ENV TOOLCHAIN="arm-gnu-toolchain-14.2.rel1-x86_64-aarch64_be-none-linux-gnu"

# Download the aarch64_be gcc toolchain
RUN curl -L "https://developer.arm.com/-/media/Files/downloads/gnu/14.2.rel1/binrel/${TOOLCHAIN}.tar.xz" -o "${TOOLCHAIN}.tar.xz"
RUN tar -xvf "${TOOLCHAIN}.tar.xz"
RUN mkdir /toolchains && mv "./${TOOLCHAIN}" /toolchains

ENV AARCH64_BE_TOOLCHAIN="/toolchains/${TOOLCHAIN}"
ENV AARCH64_BE_LIBC="${AARCH64_BE_TOOLCHAIN}/aarch64_be-none-linux-gnu/libc"

ENV CARGO_TARGET_AARCH64_BE_UNKNOWN_LINUX_GNU_LINKER="${AARCH64_BE_TOOLCHAIN}/bin/aarch64_be-none-linux-gnu-gcc"
ENV CARGO_TARGET_AARCH64_BE_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64_be -L ${AARCH64_BE_LIBC}"
ENV OBJDUMP="${AARCH64_BE_TOOLCHAIN}/bin/bin/aarch64-none-linux-gnu-objdump"
