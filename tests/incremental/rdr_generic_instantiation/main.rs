// Test that private changes in generic functions do not cause downstream
// rebuilds when the generic is instantiated in the dependent crate.
//
// - rpass1: Initial compilation
// - rpass2: Private generic helper changes body, should reuse
// - rpass3: Unused private generic added, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: generic_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate generic_dep;

fn main() {
    let x: u32 = generic_dep::generic_fn(42);
    assert_eq!(x, 42);

    let s = generic_dep::GenericStruct { value: 100u64 };
    assert_eq!(s.get(), 100);
}
