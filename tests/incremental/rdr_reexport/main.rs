// Test that re-exports work correctly with RDR. When the base crate's
// private implementation changes, crates using re-exports should reuse.
//
// - rpass1: Initial compilation
// - rpass2: Base crate's private impl changes, should reuse

//@ revisions: rpass1 rpass2
//@ compile-flags: -Z query-dep-graph
//@ aux-build: reexport_base.rs
//@ aux-build: reexport_middle.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]

extern crate reexport_middle;

fn main() {
    let thing = reexport_middle::create_thing();
    assert_eq!(thing.get(), 42);
}
