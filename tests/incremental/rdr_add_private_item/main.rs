// Test that adding private items to a dependency does not cause rebuilds.
//
// rpass1: Initial compilation
// rpass2: Auxiliary adds private items - main should be fully reused
//
// With -Z stable-crate-hash, the SVH only includes public API, so adding
// private items should not change the SVH and should not trigger rebuilds.

//@ revisions: rpass1 rpass2
//@ compile-flags: -Z query-dep-graph -Z stable-crate-hash -Z incremental-ignore-spans
//@ aux-build: lib.rs

#![feature(rustc_attrs)]

// The main module should be reused in rpass2 since the dependency's
// public API hasn't changed (only private items were added).
#![rustc_partition_reused(module = "main", cfg = "rpass2")]

extern crate lib;

pub fn use_dependency() -> i32 {
    lib::public_fn() + lib::PublicStruct { field: 10 }.field
}

fn main() {
    assert_eq!(use_dependency(), 52);
}
