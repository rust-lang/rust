#!/bin/sh
# Download musl to a repository for `musl-math-sys`

set -eux

url=git://git.musl-libc.org/musl
ref=c47ad25ea3b484e10326f933e927c0bc8cded3da
dst=crates/musl-math-sys/musl

if ! [ -d "$dst" ]; then
    git clone "$url" "$dst" --single-branch --depth=1000
fi

git -C "$dst" fetch "$url" --depth=1
git -C "$dst" checkout "$ref"
