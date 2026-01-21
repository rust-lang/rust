// Test diamond dependency pattern: A depends on B and C, both depend on D.
// When D changes privately, A, B, and C should all reuse.
//
// - rpass1: Initial compilation
// - rpass2: Base crate's private impl changes, should reuse
// - rpass3: Base crate adds another private fn, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: diamond_base.rs
//@ aux-build: diamond_left.rs
//@ aux-build: diamond_right.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate diamond_left;
extern crate diamond_right;

fn main() {
    let left = diamond_left::left_value();
    let right = diamond_right::right_value();
    assert_eq!(left, 43);
    assert_eq!(right, 44);
}
