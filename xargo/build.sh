#!/bin/bash
cd "$(readlink -e "$(dirname "$0")")"
RUSTFLAGS='-Zalways-encode-mir -Zmir-emit-validate=1' xargo build
