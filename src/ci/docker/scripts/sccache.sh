set -ex

echo Use sccache built with Azure Storage support
curl -fo /usr/local/bin/sccache \
  https://rustlangtools.blob.core.windows.net/public/stable-x86_64-unknown-linux-musl.sccache

chmod +x /usr/local/bin/sccache
