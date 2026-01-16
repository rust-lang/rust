// Test that span changes in private items of a dependency do NOT cause
// the dependent crate to be rebuilt.
//
// This is the core test for Relink, Don't Rebuild (RDR).
// The key insight is that metadata should not encode spans in a way that
// causes hash instability when only private implementation details change.
//
// Revisions:
// - rpass1: Initial compilation
// - rpass2: Dependency adds blank lines (BytePos shifts) - should REUSE
// - rpass3: Dependency adds comments in private fn - should REUSE
// - rpass4: Dependency changes private fn body - should REUSE
//
// In all revisions, the main crate should be reused because:
// 1. The public API of the dependency hasn't changed
// 2. No inlined code from the dependency is used
// 3. Span changes in private items shouldn't affect metadata hashes

//@ revisions: rpass1 rpass2 rpass3 rpass4
//@ compile-flags: -Z query-dep-graph
//@ aux-build: dep.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "bin"]

// THE KEY ASSERTION: This module should be reused in ALL subsequent revisions.
// If this fails, it means span changes in private dependency code are
// incorrectly invalidating the dependent crate's cache.
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]
#![rustc_partition_reused(module = "main", cfg = "rpass4")]

extern crate dep;

use dep::{PublicStruct, PublicTrait, public_compute};

fn main() {
    // Use the public API - none of this should cause recompilation
    // when only private spans change in the dependency.

    // Test struct construction and methods
    let s = PublicStruct::new(10);
    assert_eq!(s.value, 11); // private_transform adds 1
    assert_eq!(s.doubled(), 22); // private_double multiplies by 2

    // Test public function
    let result = public_compute(5);
    // private_transform(5) = 6
    // private_double(6) = 12
    // private_combine(6, 12) = 18
    assert_eq!(result, 18);

    // Test trait implementation
    let val: u32 = 7;
    assert_eq!(val.compute(), 7);
    assert_eq!(val.with_default(), 8); // private_transform(7) = 8

    println!("All RDR assertions passed!");
}
