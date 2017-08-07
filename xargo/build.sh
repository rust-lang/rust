#!/bin/sh
cd "$(dirname "$0")"
RUSTFLAGS='-Zalways-encode-mir -Zmir-emit-validate=1' xargo build
