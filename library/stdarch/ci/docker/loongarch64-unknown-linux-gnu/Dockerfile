FROM ubuntu:24.04
RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates wget xz-utils && \
  wget "https://github.com/loongson/build-tools/releases/download/2024.08.08/x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0.tar.xz" && \
  tar -xvf x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0.tar.xz -C / && \
  rm x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0.tar.xz && \
  wget "https://github.com/loongson/build-tools/releases/download/2024.08.08/x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0-glibc_2.40.tar.xz" && \
  tar -xvf x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0-glibc_2.40.tar.xz -C / && \
  rm x86_64-cross-tools-loongarch64-binutils_2.43-gcc_14.2.0-glibc_2.40.tar.xz && \
  wget "https://github.com/loongson/build-tools/releases/download/2023.08.08/qemu-loongarch64" && \
  install -m 755 qemu-loongarch64 /cross-tools/bin/qemu-loongarch64 && \
  rm qemu-loongarch64
ENV \
  CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_LINKER="/cross-tools/bin/loongarch64-unknown-linux-gnu-gcc" \
  CARGO_TARGET_LOONGARCH64_UNKNOWN_LINUX_GNU_RUNNER="/cross-tools/bin/qemu-loongarch64 -L /cross-tools/target -E LD_LIBRARY_PATH=/cross-tools/loongarch64-unknown-linux-gnu/lib"
