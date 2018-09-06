FROM ubuntu:18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  clang \
  cmake \
  curl \
  git \
  libc6-dev \
  make \
  python \
  xz-utils

# Install `wasm2wat`
RUN git clone --recursive https://github.com/WebAssembly/wabt
RUN make -C wabt -j$(nproc)
ENV PATH=$PATH:/wabt/bin

# Install `wasm-bindgen-test-runner`
RUN curl -L https://github.com/rustwasm/wasm-bindgen/releases/download/0.2.19/wasm-bindgen-0.2.19-x86_64-unknown-linux-musl.tar.gz \
  | tar xzf -
ENV PATH=$PATH:/wasm-bindgen-0.2.19-x86_64-unknown-linux-musl
ENV CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner

# Install `node`
RUN curl https://nodejs.org/dist/v10.8.0/node-v10.8.0-linux-x64.tar.xz | tar xJf -
ENV PATH=$PATH:/node-v10.8.0-linux-x64/bin

# We use a shim linker that removes `--strip-debug` when passed to LLD. While
# this typically results in invalid debug information in release mode it doesn't
# result in an invalid names section which is what we're interested in.
COPY lld-shim.rs /
ENV CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER=/tmp/lld-shim

# Rustc isn't available until this container starts, so defer compilation of the
# shim.
ENTRYPOINT /rust/bin/rustc /lld-shim.rs -o /tmp/lld-shim && exec bash "$@"
