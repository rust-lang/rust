FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  lib32stdc++6

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      tar xJf - -C /usr/local/bin --strip-components=1

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

WORKDIR /tmp
COPY build-emscripten.sh /tmp/
RUN ./build-emscripten.sh
ENV PATH=$PATH:/tmp/emsdk_portable
ENV PATH=$PATH:/tmp/emsdk_portable/clang/tag-e1.37.1/build_tag-e1.37.1_32/bin
ENV PATH=$PATH:/tmp/emsdk_portable/node/4.1.1_32bit/bin
ENV PATH=$PATH:/tmp/emsdk_portable/emscripten/tag-1.37.1
ENV EMSCRIPTEN=/tmp/emsdk_portable/emscripten/tag-1.37.1

ENV RUST_CONFIGURE_ARGS --target=asmjs-unknown-emscripten

# Run `emcc` first as it's got a prompt and doesn't actually do anything, after
# that's done with do the real build.
ENV SCRIPT emcc && \
      python2.7 ../x.py test --target asmjs-unknown-emscripten

