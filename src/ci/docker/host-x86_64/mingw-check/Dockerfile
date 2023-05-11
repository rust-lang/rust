FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  python3-pip \
  python3-pkg-resources \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://nodejs.org/dist/v16.9.0/node-v16.9.0-linux-x64.tar.xz | tar -xJ
ENV PATH="/node-v16.9.0-linux-x64/bin:${PATH}"
ENV RUST_CONFIGURE_ARGS="--set rust.validate-mir-opts=3"

# Install es-check
# Pin its version to prevent unrelated CI failures due to future es-check versions.
RUN npm install es-check@6.1.1 eslint@8.6.0 -g

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/reuse-requirements.txt /tmp/
RUN pip3 install --no-deps --no-cache-dir --require-hashes -r /tmp/reuse-requirements.txt

COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/
COPY host-x86_64/mingw-check/validate-error-codes.sh /scripts/

ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1
ENV SCRIPT python3 ../x.py --stage 2 test src/tools/expand-yaml-anchors && \
           python3 ../x.py check --target=i686-pc-windows-gnu --host=i686-pc-windows-gnu && \
           python3 ../x.py build --stage 0 src/tools/build-manifest && \
           python3 ../x.py test --stage 0 src/tools/compiletest && \
           python3 ../x.py test --stage 0 core alloc std test proc_macro && \
           # Build both public and internal documentation.
           RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 compiler && \
           RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 library/test && \
           /scripts/validate-toolstate.sh && \
           /scripts/validate-error-codes.sh && \
           reuse lint && \
           # Runs checks to ensure that there are no ES5 issues in our JS code.
           es-check es6 ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/librustdoc/html/static/.eslintrc.js ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/tools/rustdoc-js/.eslintrc.js ../src/tools/rustdoc-js/tester.js && \
           eslint -c ../src/tools/rustdoc-gui/.eslintrc.js ../src/tools/rustdoc-gui/tester.js
