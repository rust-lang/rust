// Test that private changes in a derive macro crate do not cause
// downstream crates to rebuild.
//
// - rpass1: Initial compilation
// - rpass2: Macro's private helper changes, should reuse
// - rpass3: Macro adds more private items, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ proc-macro: derive_helper.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

#[macro_use]
extern crate derive_helper;

#[derive(RdrTestDerive)]
pub struct TestStruct;

fn main() {
    let value = TestStruct::derived_value();
    assert_eq!(value, 42);
}
