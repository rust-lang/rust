// This test verifies that span hashing is stable across incremental
// compilation sessions for cross-crate dependencies.
//
// For RDR (Relink, Don't Rebuild), it's critical that:
// 1. Span hashes use file:line:column rather than raw byte offsets
// 2. Cross-crate spans are hashed consistently
// 3. Changing unrelated code doesn't invalidate span hashes
//
// rpass1: Initial compilation
// rpass2: Recompile with no changes - all modules should be reused
// rpass3: Recompile again - still should reuse everything

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph -g
//@ aux-build: hash_lib.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

// All modules should be reused since neither the source nor the
// auxiliary crate changed between revisions.
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate hash_lib;

use hash_lib::{StableStruct, generic_stable, stable_span_fn};

fn main() {
    // Test inline function with stable span
    let val = stable_span_fn();
    assert_eq!(val, 42);

    // Test generic function monomorphization
    let x: u32 = generic_stable(21);
    assert_eq!(x, 21);

    let y: i64 = generic_stable(100);
    assert_eq!(y, 100);

    // Test struct with derived traits (derives embed spans)
    let s1 = StableStruct::new(42);
    let s2 = s1.clone();
    assert_eq!(s1, s2);

    // Debug formatting also uses spans from derive
    let debug_str = format!("{:?}", s1);
    assert!(debug_str.contains("42"));
}
