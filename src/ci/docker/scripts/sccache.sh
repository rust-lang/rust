set -ex

curl -fo /usr/local/bin/sccache \
  https://ci-mirrors.rust-lang.org/rustc/2018-04-02-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
