// Test that const generics with private const values work correctly with RDR.
//
// - rpass1: Initial compilation
// - rpass2: Private const unchanged (same value), should reuse
// - rpass3: Unused private const added, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: const_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate const_dep;

fn main() {
    let arr = const_dep::make_default();
    assert_eq!(arr.len(), 4);

    let custom: const_dep::FixedArray<8> = const_dep::FixedArray::new();
    assert_eq!(custom.len(), 8);
}
