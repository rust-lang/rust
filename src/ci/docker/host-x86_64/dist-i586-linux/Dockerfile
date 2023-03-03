FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  xz-utils \
  sudo \
  gdb \
  patch \
  libssl-dev \
  pkg-config

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng-1.24.sh /scripts/
RUN sh /scripts/crosstool-ng-1.24.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY host-x86_64/dist-i586-linux/i586-linux-gnu.config \
     host-x86_64/dist-i586-linux/build-i586-gnu-toolchain.sh \
     /tmp/
RUN su rustbuild -c ./build-i586-gnu-toolchain.sh
ENV PATH=$PATH:/x-tools/i586-unknown-linux-gnu/bin
ENV \
    CC_i586_unknown_linux_gnu=i586-unknown-linux-gnu-gcc \
    AR_i586_unknown_linux_gnu=i586-unknown-linux-gnu-ar

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
        --enable-full-tools \
        --disable-docs

# Newer binutils broke things on some vms/distros (i.e., linking against
# unknown relocs disabled by the following flag), so we need to go out of our
# way to produce "super compatible" binaries.
#
# See: https://github.com/rust-lang/rust/issues/34978
ENV CFLAGS_i586_unknown_linux_gnu=-Wa,-mrelax-relocations=no

ENV HOSTS=i586-unknown-linux-gnu
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS

