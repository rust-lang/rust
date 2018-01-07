FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  xz-utils \
  sudo \
  gdb \
  patch \
  libssl-dev \
  pkg-config

WORKDIR /build/
COPY scripts/musl.sh /build/
RUN CC=gcc CFLAGS="-m32 -fPIC -Wa,-mrelax-relocations=no" \
    CXX=g++ CXXFLAGS="-m32 -Wa,-mrelax-relocations=no" \
    bash musl.sh i686 --target=i686 && \
    rm -rf /build

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
      --target=i686-unknown-linux-musl,i586-unknown-linux-gnu \
      --musl-root-i686=/musl-i686 \
      --enable-extended

# Newer binutils broke things on some vms/distros (i.e., linking against
# unknown relocs disabled by the following flag), so we need to go out of our
# way to produce "super compatible" binaries.
#
# See: https://github.com/rust-lang/rust/issues/34978
ENV CFLAGS_i686_unknown_linux_musl=-Wa,-mrelax-relocations=no
ENV CFLAGS_i586_unknown_linux_gnu=-Wa,-mrelax-relocations=no

ENV SCRIPT \
      python2.7 ../x.py test \
          --target i686-unknown-linux-musl \
          --target i586-unknown-linux-gnu \
          && \
      python2.7 ../x.py dist \
          --target i686-unknown-linux-musl \
          --target i586-unknown-linux-gnu
