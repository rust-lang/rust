set -ex

curl -fo /usr/local/bin/sccache \
  https://rust-lang-ci-mirrors.s3-us-west-1.amazonaws.com/rustc/2018-04-02-sccache-x86_64-unknown-linux-musl

chmod +x /usr/local/bin/sccache
