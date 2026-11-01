FROM ubuntu:25.10

RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libc6-dev \
  ca-certificates \
  curl \
  zstd \
  file \
  make \
  libc++1 \
  libglib2.0-0t64 \
  libunwind-20 \
  liburing2 \
  llvm

# The Hexagon toolchain requires libc++ and libunwind at runtime - create symlinks from versioned files
RUN cd /usr/lib/x86_64-linux-gnu && \
    for f in libc++.so.1.0.*; do ln -sf "$f" libc++.so.1; done && \
    for f in libc++abi.so.1.0.*; do ln -sf "$f" libc++abi.so.1; done && \
    for f in libunwind.so.1.0.*; do ln -sf "$f" libunwind.so.1; done

# Download and install the Hexagon cross toolchain from
# https://github.com/quic/toolchain_for_hexagon/releases/tag/v21.1.8
# Includes clang cross-compiler, musl sysroot, and qemu-hexagon.
#
# The tarball contains directories with restrictive (0700) permissions.
# In rootless Podman, chmod fails on tar-extracted files within the same
# layer due to overlayfs limitations in user namespaces. Splitting into
# two RUN steps lets chmod work via overlayfs copy-up from the lower layer.
RUN curl -L -o /tmp/hexagon-toolchain.tar.zst \
    https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/21.1.8/clang+llvm-21.1.8-cross-hexagon-unknown-linux-musl.tar.zst && \
    mkdir -p /opt/hexagon-toolchain && \
    cd /opt/hexagon-toolchain && \
    (unzstd -c /tmp/hexagon-toolchain.tar.zst | tar -xf - --strip-components=2 --no-same-permissions || true) && \
    rm /tmp/hexagon-toolchain.tar.zst
RUN find /opt/hexagon-toolchain -type d -exec chmod a+rx {} + 2>/dev/null; \
    find /opt/hexagon-toolchain -type f -exec chmod a+r {} + 2>/dev/null; \
    find /opt/hexagon-toolchain -type f -perm /111 -exec chmod a+rx {} + 2>/dev/null; \
    /opt/hexagon-toolchain/bin/hexagon-unknown-linux-musl-clang --version

ENV PATH="/opt/hexagon-toolchain/bin:${PATH}" \
    CARGO_TARGET_HEXAGON_UNKNOWN_LINUX_MUSL_LINKER=hexagon-unknown-linux-musl-clang \
    CARGO_TARGET_HEXAGON_UNKNOWN_LINUX_MUSL_RUNNER="qemu-hexagon -L /opt/hexagon-toolchain/target/hexagon-unknown-linux-musl" \
    CARGO_UNSTABLE_BUILD_STD_FEATURES=llvm-libunwind \
    OBJDUMP=llvm-objdump
