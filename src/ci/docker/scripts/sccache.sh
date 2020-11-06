#!/bin/sh
set -ex

case "$(uname -m)" in
    x86_64)
        url="https://ci-mirrors.rust-lang.org/rustc/2018-04-02-sccache-x86_64-unknown-linux-musl"
        ;;
    aarch64)
        url="https://ci-mirrors.rust-lang.org/rustc/2019-12-17-sccache-aarch64-unknown-linux-gnu"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fo /usr/local/bin/sccache "${url}"
chmod +x /usr/local/bin/sccache
