#!/bin/sh

# ignore-tidy-linelength

set -ex

case "$(uname -m)" in
    x86_64)
        url="https://github.com/mozilla/sccache/releases/download/v0.3.3/sccache-v0.3.3-x86_64-unknown-linux-musl.tar.gz"
        ;;
    aarch64)
        url="https://ci-mirrors.rust-lang.org/rustc/2021-08-25-sccache-v0.2.15-aarch64-unknown-linux-musl"
        ;;
    *)
        echo "unsupported architecture: $(uname -m)"
        exit 1
esac

curl -fLo sccache.tar.gz "${url}"
tar zxvf sccache.tar.gz --wildcards --no-anchored 'sccache' --strip-components=1
cp sccache /usr/local/bin/sccache
chmod +x /usr/local/bin/sccache
