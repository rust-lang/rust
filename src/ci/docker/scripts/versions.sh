#!/usr/bin/env bash
# ignore-tidy-linelength

# To keep docker / non-docker builds in sync

# renovate: datasource=github-releases depName=llvm/llvm-project versioning=semver-coerced extractVersion=^llvmorg-(?<version>\d+\.\d+\.\d+(?:.*))
export LLVM_VERSION=21.1.0-rc2
