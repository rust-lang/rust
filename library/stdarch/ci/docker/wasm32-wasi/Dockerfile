FROM rust:1.50.0

# Install wasmtime from source for now while the `experimental_x64` feature is
# not yet the default. (it's not actually that experimental at the time of this
# writing, wasmtime should switch defaults soon and the backend this enables has
# better support for simd instructions)
RUN \
  CARGO_INCREMENTAL=0 \
  CARGO_PROFILE_DEV_DEBUGINFO=0 \
  cargo install wasmtime-cli --features experimental_x64 --debug --vers 0.25.0 --locked

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  clang

COPY --from=0 /usr/local/cargo/bin/wasmtime /usr/local/bin/wasmtime

ENV CARGO_TARGET_WASM32_WASI_RUNNER="wasmtime \
  --enable-simd \
  --enable-threads \
  --opt-level 0 \
  --mapdir .::/checkout/target/wasm32-wasi/release/deps \
  --"
