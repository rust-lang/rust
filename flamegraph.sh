#!/bin/bash
source config.sh

# These files grow really big (~1.4G) because of the sample frequency
rm perf.data* || true

# Profile compiling libcore
perf record -F 9000 --call-graph dwarf \
    -- $RUSTC --crate-type lib build_sysroot/sysroot_src/src/libcore/lib.rs --crate-name core

# Generate the flamegraph
perf script | ../FlameGraph/stackcollapse-perf.pl | grep cranelift | ../FlameGraph/flamegraph.pl > abc.svg
