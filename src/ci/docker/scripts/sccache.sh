#!/bin/sh

# ignore-tidy-linelength

set -ex

case "$(uname -m)" in
    x86_64)
        url="https://ci-mirrors.rust-lang.org/rustc/2025-01-07-sccache-v0.9.1-x86_64-unknown-linux-musl"
        ;;
    aarch64)
        url="https://ci-mirrors.rust-lang.org/rustc/2025-01-07-sccache-v0.9.1-aarch64-unknown-linux-musl"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fo /usr/local/bin/sccache "${url}"
chmod +x /usr/local/bin/sccache
