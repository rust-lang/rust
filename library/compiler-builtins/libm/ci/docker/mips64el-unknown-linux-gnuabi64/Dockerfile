FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    gcc-mips64el-linux-gnuabi64 \
    libc6-dev \
    libc6-dev-mips64el-cross \
    qemu-user-static
ENV CARGO_TARGET_MIPS64EL_UNKNOWN_LINUX_GNUABI64_LINKER=mips64el-linux-gnuabi64-gcc \
    CARGO_TARGET_MIPS64EL_UNKNOWN_LINUX_GNUABI64_RUNNER=qemu-mips64el-static \
    CC_mips64el_unknown_linux_gnuabi64=mips64el-linux-gnuabi64-gcc \
    QEMU_LD_PREFIX=/usr/mips64el-linux-gnuabi64 \
    RUST_TEST_THREADS=1
