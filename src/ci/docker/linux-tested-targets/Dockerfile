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
  patch

WORKDIR /build/
COPY musl-libunwind-patch.patch build-musl.sh /build/
RUN sh /build/build-musl.sh && rm -rf /build

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      tar xJf - -C /usr/local/bin --strip-components=1

ENV RUST_CONFIGURE_ARGS \
      --target=x86_64-unknown-linux-musl,i686-unknown-linux-musl,i586-unknown-linux-gnu \
      --musl-root-x86_64=/musl-x86_64 \
      --musl-root-i686=/musl-i686

# FIXME should also test i686-unknown-linux-musl
ENV SCRIPT \
      python2.7 ../x.py test \
          --target x86_64-unknown-linux-musl \
          --target i586-unknown-linux-gnu \
          && \
      python2.7 ../x.py dist \
          --target x86_64-unknown-linux-musl \
          --target i686-unknown-linux-musl \
          --target i586-unknown-linux-gnu
