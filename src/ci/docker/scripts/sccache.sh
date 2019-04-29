set -ex

curl -fo /usr/local/bin/sccache \
  https://s3-us-west-1.amazonaws.com/rust-lang-ci2/rust-ci-mirror/2018-04-02-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
