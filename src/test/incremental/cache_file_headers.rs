// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test case makes sure that the compiler does not try to re-use anything
// from the incremental compilation cache if the cache was produced by a
// different compiler version. This is tested by artificially forcing the
// emission of a different compiler version in the header of rpass1 artifacts,
// and then making sure that the only object file of the test program gets
// re-translated although the program stays unchanged.

// The `l33t haxx0r` Rust compiler is known to produce incr. comp. artifacts
// that are outrageously incompatible with just about anything, even itself:
//[rpass1] rustc-env:RUSTC_FORCE_INCR_COMP_ARTIFACT_HEADER="l33t haxx0r rustc 2.1 LTS"

// revisions:rpass1 rpass2

#![feature(rustc_attrs)]
#![rustc_partition_translated(module="cache_file_headers", cfg="rpass2")]

fn main() {
    // empty
}
