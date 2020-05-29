FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc libc6-dev ca-certificates

ENV CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=true
