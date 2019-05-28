set -ex

curl -fo /usr/local/bin/sccache \
  https://rust-lang-ci2.s3.amazonaws.com/rust-ci-mirror/2018-04-02-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
