// Test that platform-specific private code changes do not cause
// downstream rebuilds.
//
// - rpass1: Initial compilation
// - rpass2: Platform-specific private fn changes, should reuse
// - rpass3: Extra private fn added, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph
//@ aux-build: cfg_dep.rs
//@ edition: 2024
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate cfg_dep;

fn main() {
    let platform = cfg_dep::get_platform();
    assert!(!platform.is_empty());
}
