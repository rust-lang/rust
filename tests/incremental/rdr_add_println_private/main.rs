// Test that adding println!/eprintln! to private functions in a dependency
// does NOT cause the dependent crate to be rebuilt.
//
// This is a key RDR test because:
// 1. println! is a macro that expands to code with many spans
// 2. The expanded code includes format strings, arguments, etc.
// 3. All of this should be invisible to dependent crates
//
// Revisions:
// - rpass1: No println in private functions
// - rpass2: Add println!/eprintln! to private functions - should REUSE
// - rpass3: Add more println! statements - should REUSE

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: dep.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "bin"]

// Main should be reused - adding println to private fns shouldn't affect us
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate dep;

fn main() {
    let result = dep::public_fn(41);
    assert_eq!(result, 42);

    let s = dep::PublicStruct { value: 21 };
    assert_eq!(s.compute(), 42);
}
