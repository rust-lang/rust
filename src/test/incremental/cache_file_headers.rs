// This test case makes sure that the compiler does not try to re-use anything
// from the incremental compilation cache if the cache was produced by a
// different compiler version. This is tested by artificially forcing the
// emission of a different compiler version in the header of rpass1 artifacts,
// and then making sure that the only object file of the test program gets
// re-codegened although the program stays unchanged.

// The `l33t haxx0r` Rust compiler is known to produce incr. comp. artifacts
// that are outrageously incompatible with just about anything, even itself:
//[rpass1] rustc-env:RUSTC_FORCE_RUSTC_VERSION="l33t haxx0r rustc 2.1 LTS"

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![rustc_partition_codegened(module="cache_file_headers", cfg="rpass2")]

fn main() {
    // empty
}
