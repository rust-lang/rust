#!/bin/sh

# ignore-tidy-linelength

set -ex

case "$(uname -m)" in
    x86_64)
        url="https://ci-mirrors.rust-lang.org/rustc/2022-12-08-sccache-v0.3.3-x86_64-unknown-linux-musl"
        ;;
    aarch64)
        # no aarch64 release for 0.3.3, so use this
        url="https://ci-mirrors.rust-lang.org/rustc/2022-11-13-sccache-v0.3.1-aarch64-unknown-linux-musl"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fo /usr/local/bin/sccache "${url}"
chmod +x /usr/local/bin/sccache
