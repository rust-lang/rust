// Test that adding private functions to a dependency does NOT cause
// the dependent crate to be rebuilt.
//
// This is a core RDR (Relink, Don't Rebuild) test.
// Adding private items should not affect the public API hash.
//
// Revisions:
// - rpass1: Initial compilation
// - rpass2: Dependency adds one private function - should REUSE
// - rpass3: Dependency adds more private items - should REUSE

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: dep.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "bin"]

// Main should be reused when only private items are added to dependency
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate dep;

fn main() {
    let result = dep::public_fn(41);
    assert_eq!(result, 42);

    let s = dep::PublicStruct { value: 100 };
    assert_eq!(s.value, 100);
}
