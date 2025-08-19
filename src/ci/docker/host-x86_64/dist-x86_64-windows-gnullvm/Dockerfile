FROM ubuntu:24.04

WORKDIR /build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    make \
    ninja-build \
    python3 \
    xz-utils

ENV ARCH='i686 x86_64'
COPY host-x86_64/dist-x86_64-windows-gnullvm/install-llvm-mingw.sh /build
RUN ./install-llvm-mingw.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV CC_i686_pc_windows_gnullvm=i686-w64-mingw32-clang \
    CC_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang \
    CXX_x86_64_pc_windows_gnullvm=x86_64-w64-mingw32-clang++

ENV HOST=x86_64-pc-windows-gnullvm
ENV TARGETS=i686-pc-windows-gnullvm,x86_64-pc-windows-gnullvm

# We are bootstrapping this target and cannot use previously built artifacts.
# Without this option Clang is given `"-I/checkout/obj/build/aarch64-pc-windows-gnullvm/ci-llvm/include"`
# despite no such directory existing:
# $ ls obj/dist-windows-gnullvm/build/aarch64-pc-windows-gnullvm/ -1
# llvm
# stage2
ENV NO_DOWNLOAD_CI_LLVM 1

ENV RUST_CONFIGURE_ARGS \
    --enable-extended \
    --enable-profiler \
    --enable-sanitizers \
    --disable-docs \
    --set llvm.download-ci-llvm=false \
    --set rust.llvm-tools=false
# LLVM cross tools are not installed into expected location so copying fails.
# Probably will solve itself once these targets can host themselves on Windows.
# --enable-full-tools \

ENV SCRIPT python3 ../x.py dist --host $HOST --target $TARGETS
