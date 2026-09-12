#!/bin/sh

# ignore-tidy-file-linelength

set -ex

case "$(uname -m)" in
    x86_64)
        url="https://ci-mirrors.rust-lang.org/rustc/2026-06-19-sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz"
        ;;
    aarch64)
        url="https://ci-mirrors.rust-lang.org/rustc/2026-06-19-sccache-v0.16.0-aarch64-unknown-linux-musl.tar.gz"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fo sccache.tar.gz "${url}"
tar -xvf sccache.tar.gz --strip-components 1
mv sccache /usr/local/bin/sccache
chmod +x /usr/local/bin/sccache
