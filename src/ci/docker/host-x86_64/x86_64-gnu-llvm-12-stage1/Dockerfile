FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  gcc-multilib \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  gdb \
  llvm-12-tools \
  llvm-12-dev \
  libedit-dev \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  xz-utils \
  nodejs

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# using llvm-link-shared due to libffi issues -- see #34486
ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-12 \
      --enable-llvm-link-shared \
      --set rust.thin-lto-import-instr-limit=10

# Some tests use --stage 1 to mean "build using stage1" and some using it to mean "run using stage 1".
# Until we straighten that out, special-case the ones that mean "build with stage1".
ENV SCRIPT python2.7 ../x.py test --exclude run-make-fulldeps --exclude error_index_generator \
            --exclude linkchecker --stage 1 --exclude src/tools/rustdoc --exclude rustdoc-json-types \
            --exclude tidy --exclude ui-fulldeps && \
           # Intentionally excludes *-fulldeps, error-index, and linkchecker tests, which aren't guaranteed to work in stage 1.
           python2.7 ../x.py test --stage 0 src/tools/rustdoc rustdoc-json-types && \
           # Run the `mir-opt` tests again but this time for a 32-bit target.
           # This enforces that tests using `// EMIT_MIR_FOR_EACH_BIT_WIDTH` have
           # both 32-bit and 64-bit outputs updated by the PR author, before
           # the PR is approved and tested for merging.
           # It will also detect tests lacking `// EMIT_MIR_FOR_EACH_BIT_WIDTH`,
           # despite having different output on 32-bit vs 64-bit targets.
           python2.7 ../x.py --stage 1 test src/test/mir-opt \
                             --host='' --target=i686-unknown-linux-gnu && \
           # Run the UI test suite again, but in `--pass=check` mode
           #
           # This is intended to make sure that both `--pass=check` continues to
           # work.
           #
           python2.7 ../x.py --stage 1 test src/test/ui --pass=check \
                             --host='' --target=i686-unknown-linux-gnu && \
           # Run tidy at the very end, after all the other tests.
           python2.7 ../x.py --stage 1 test src/tools/tidy
