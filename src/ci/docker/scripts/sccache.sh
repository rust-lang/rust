#!/bin/sh

# ignore-tidy-linelength

set -ex

case "$(uname -m)" in
    x86_64)
        url="https://ci-mirrors.rust-lang.org/rustc/2021-08-24-sccache-v0.2.15-x86_64-unknown-linux-musl"
        ;;
    aarch64)
        url="https://ci-mirrors.rust-lang.org/rustc/2021-08-25-sccache-v0.2.15-aarch64-unknown-linux-musl"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fo /usr/local/bin/sccache "${url}"
chmod +x /usr/local/bin/sccache
