FROM ubuntu:23.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    ninja-build \
    file \
    curl \
    ca-certificates \
    python3 \
    git \
    cmake \
    sudo \
    gdb \
    libssl-dev \
    pkg-config \
    xz-utils \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/ohos-sdk.sh /scripts/
RUN sh /scripts/ohos-sdk.sh

COPY scripts/ohos/aarch64-unknown-linux-ohos-clang.sh /usr/local/bin/
COPY scripts/ohos/aarch64-unknown-linux-ohos-clang++.sh /usr/local/bin/
COPY scripts/ohos/armv7-unknown-linux-ohos-clang.sh /usr/local/bin/
COPY scripts/ohos/armv7-unknown-linux-ohos-clang++.sh /usr/local/bin/
COPY scripts/ohos/x86_64-unknown-linux-ohos-clang.sh /usr/local/bin/
COPY scripts/ohos/x86_64-unknown-linux-ohos-clang++.sh /usr/local/bin/

# env
ENV TARGETS=aarch64-unknown-linux-ohos
ENV TARGETS=$TARGETS,armv7-unknown-linux-ohos
ENV TARGETS=$TARGETS,x86_64-unknown-linux-ohos

ENV \
    CC_aarch64_unknown_linux_ohos=/usr/local/bin/aarch64-unknown-linux-ohos-clang.sh \
    AR_aarch64_unknown_linux_ohos=/opt/ohos-sdk/native/llvm/bin/llvm-ar \
    CXX_aarch64_unknown_linux_ohos=/usr/local/bin/aarch64-unknown-linux-ohos-clang++.sh
ENV \
    CC_armv7_unknown_linux_ohos=/usr/local/bin/armv7-unknown-linux-ohos-clang.sh \
    AR_armv7_unknown_linux_ohos=/opt/ohos-sdk/native/llvm/bin/llvm-ar \
    CXX_armv7_unknown_linux_ohos=/usr/local/bin/armv7-unknown-linux-ohos-clang++.sh
ENV \
    CC_x86_64_unknown_linux_ohos=/usr/local/bin/x86_64-unknown-linux-ohos-clang.sh \
    AR_x86_64_unknown_linux_ohos=/opt/ohos-sdk/native/llvm/bin/llvm-ar \
    CXX_x86_64_unknown_linux_ohos=/usr/local/bin/x86_64-unknown-linux-ohos-clang++.sh

ENV RUST_CONFIGURE_ARGS \
    --enable-profiler \
    --disable-docs

ENV SCRIPT python3 ../x.py dist --host='' --target $TARGETS

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
