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
COPY musl-libunwind-patch.patch build-musl.sh /build/
RUN sh /build/build-musl.sh && rm -rf /build

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-02-24-sccache-x86_64-unknown-linux-gnu && \
      chmod +x /usr/local/bin/sccache

ENV RUST_CONFIGURE_ARGS \
      --target=x86_64-unknown-linux-musl,i686-unknown-linux-musl,i586-unknown-linux-gnu \
      --musl-root-x86_64=/musl-x86_64 \
      --musl-root-i686=/musl-i686

# Newer binutils broke things on some vms/distros (i.e., linking against
# unknown relocs disabled by the following flag), so we need to go out of our
# way to produce "super compatible" binaries.
#
# See: https://github.com/rust-lang/rust/issues/34978
ENV CFLAGS_i686_unknown_linux_gnu=-Wa,-mrelax-relocations=no \
    CFLAGS_x86_64_unknown_linux_gnu=-Wa,-mrelax-relocations=no

ENV SCRIPT \
      python2.7 ../x.py test \
          --target x86_64-unknown-linux-musl \
          --target i686-unknown-linux-musl \
          --target i586-unknown-linux-gnu \
          && \
      python2.7 ../x.py dist \
          --target x86_64-unknown-linux-musl \
          --target i686-unknown-linux-musl \
          --target i586-unknown-linux-gnu
