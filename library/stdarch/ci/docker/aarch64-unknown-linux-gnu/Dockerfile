FROM ubuntu:25.10
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  ca-certificates \
  libc6-dev \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  libc6-dev-arm64-cross \
  qemu-user \
  make \
  file \
  xz-utils \
  wget

RUN wget https://mirrors.edge.kernel.org/pub/tools/llvm/files/llvm-22.1.4-x86_64.tar.gz -O llvm.tar.xz
RUN mkdir llvm
RUN tar -xvf llvm.tar.xz --strip-components=1 -C llvm

ENV PATH="/llvm/bin:$PATH"

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu" \
    OBJDUMP=aarch64-linux-gnu-objdump
