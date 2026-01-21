// Test that blanket impl with private trait bounds work correctly with RDR.
//
// - rpass1: Initial compilation
// - rpass2: Private trait impl changes, should reuse
// - rpass3: Another private trait added, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: blanket_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate blanket_dep;

use blanket_dep::PublicTrait;

fn main() {
    let x: u32 = 0;
    assert_eq!(x.public_method(), 42);

    let s = String::new();
    assert_eq!(s.public_method(), 42);
}
