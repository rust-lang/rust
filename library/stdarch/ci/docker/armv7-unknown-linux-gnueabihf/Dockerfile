FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  ca-certificates \
  libc6-dev \
  gcc-arm-linux-gnueabihf \
  libc6-dev-armhf-cross \
  qemu-user \
  make \
  file \
  wget

RUN wget https://mirrors.edge.kernel.org/pub/tools/llvm/files/llvm-22.1.4-x86_64.tar.gz -O llvm.tar.xz
RUN mkdir llvm
RUN tar -xvf llvm.tar.xz --strip-components=1 -C llvm

ENV PATH="/llvm/bin:$PATH"

ENV CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_LINKER=arm-linux-gnueabihf-gcc \
    CARGO_TARGET_ARMV7_UNKNOWN_LINUX_GNUEABIHF_RUNNER="qemu-arm -cpu max -L /usr/arm-linux-gnueabihf" \
    OBJDUMP=arm-linux-gnueabihf-objdump
