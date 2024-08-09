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
RUN npm install es-check@6.1.1 eslint@8.6.0 -g

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/mingw-check/reuse-requirements.txt /tmp/
RUN pip3 install --no-deps --no-cache-dir --require-hashes -r /tmp/reuse-requirements.txt

COPY host-x86_64/mingw-check/check-default-config-profiles.sh /scripts/
COPY host-x86_64/mingw-check/validate-toolstate.sh /scripts/
COPY host-x86_64/mingw-check/validate-error-codes.sh /scripts/

ENV RUN_CHECK_WITH_PARALLEL_QUERIES 1

# Check library crates on all tier 1 targets.
# We disable optimized compiler built-ins because that requires a C toolchain for the target.
# We also skip the x86_64-unknown-linux-gnu target as it is well-tested by other jobs.
ENV SCRIPT \
           # `core::builder::tests::ci_rustc_if_unchanged_logic` bootstrap test covers the `rust.download-rustc=if-unchanged` logic.
           # Here we are adding a dummy commit on compiler and running that test to ensure when there is a change on the compiler,
           # we never download ci rustc with `rust.download-rustc=if-unchanged` option.
           echo \"\" >> ../compiler/rustc/src/main.rs && \
           git config --global user.email \"dummy@dummy.com\" && \
           git config --global user.name \"dummy\" && \
           git add ../compiler/rustc/src/main.rs && \
           git commit -m \"test commit for rust.download-rustc=if-unchanged logic\" && \
           DISABLE_CI_RUSTC_IF_INCOMPATIBLE=0 python3 ../x.py test bootstrap -- core::builder::tests::ci_rustc_if_unchanged_logic && \
           # Revert the dummy commit
           git reset --hard HEAD~1 && \

           python3 ../x.py check --stage 0 --set build.optimized-compiler-builtins=false core alloc std --target=aarch64-unknown-linux-gnu,i686-pc-windows-msvc,i686-unknown-linux-gnu,x86_64-apple-darwin,x86_64-pc-windows-gnu,x86_64-pc-windows-msvc && \
           /scripts/check-default-config-profiles.sh && \
           python3 ../x.py check --target=i686-pc-windows-gnu --host=i686-pc-windows-gnu && \
           python3 ../x.py clippy bootstrap -Dwarnings && \
           python3 ../x.py clippy compiler library -Aclippy::all -Dclippy::correctness && \
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
           # Runs checks to ensure that there are no issues in our JS code.
           es-check es2019 ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/librustdoc/html/static/.eslintrc.js ../src/librustdoc/html/static/js/*.js && \
           eslint -c ../src/tools/rustdoc-js/.eslintrc.js ../src/tools/rustdoc-js/tester.js && \
           eslint -c ../src/tools/rustdoc-gui/.eslintrc.js ../src/tools/rustdoc-gui/tester.js
