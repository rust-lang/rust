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

ENV RUST_CONFIGURE_ARGS="--set rust.validate-mir-opts=3"

COPY scripts/nodejs.sh /scripts/
RUN sh /scripts/nodejs.sh /node
ENV PATH="/node/bin:${PATH}"

# Install es-check
# Pin its version to prevent unrelated CI failures due to future es-check versions.
RUN npm install es-check@6.1.1 eslint@8.6.0 typescript@5.7.3 -g

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/reuse-requirements.txt /tmp/
RUN pip3 install --no-deps --no-cache-dir --require-hashes -r /tmp/reuse-requirements.txt

COPY host-x86_64/mingw-check/check-default-config-profiles.sh /scripts/
COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/
COPY host-x86_64/mingw-check/validate-error-codes.sh /scripts/

# Check library crates on all tier 1 targets.
# We disable optimized compiler built-ins because that requires a C toolchain for the target.
# We also skip the x86_64-unknown-linux-gnu target as it is well-tested by other jobs.
ENV SCRIPT \
           python3 ../x.py check --stage 0 --set build.optimized-compiler-builtins=false core alloc std --target=aarch64-unknown-linux-gnu,i686-pc-windows-msvc,i686-unknown-linux-gnu,x86_64-apple-darwin,x86_64-pc-windows-gnu,x86_64-pc-windows-msvc && \
           /scripts/check-default-config-profiles.sh && \
           python3 ../x.py check compiletest --set build.compiletest-use-stage0-libtest=true && \
           python3 ../x.py check --target=x86_64-pc-windows-gnu --host=x86_64-pc-windows-gnu && \
           python3 ../x.py clippy ci && \
           python3 ../x.py build --stage 0 src/tools/build-manifest && \
           python3 ../x.py test --stage 0 src/tools/compiletest && \
           python3 ../x.py test --stage 0 core alloc std test proc_macro && \
           # Build both public and internal documentation.
           RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 library && \
           mkdir -p /checkout/obj/staging/doc && \
           cp -r build/x86_64-unknown-linux-gnu/doc /checkout/obj/staging && \
           RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 compiler && \
           RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 library/test && \
           /scripts/validate-toolstate.sh && \
           /scripts/validate-error-codes.sh && \
           reuse --include-submodules lint && \
           python3 ../x.py test collect-license-metadata && \
           # Runs checks to ensure that there are no issues in our JS code.
           es-check es2019 ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/librustdoc/html/static/.eslintrc.js ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/tools/rustdoc-js/.eslintrc.js ../src/tools/rustdoc-js/tester.js && \
           eslint -c ../src/tools/rustdoc-gui/.eslintrc.js ../src/tools/rustdoc-gui/tester.js && \
           tsc --project ../src/librustdoc/html/static/js/tsconfig.json
