// Test that RDR works correctly with -Zshare-generics.
//
// With share-generics, monomorphized generic instances are shared across crates
// rather than being duplicated. This test verifies that private changes in a
// dependency don't trigger rebuilds even when share-generics is enabled.
//
// - rpass1: Initial compilation
// - rpass2: Private generic helper changes body, should reuse
// - rpass3: Private non-generic function changes, should reuse

//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Z query-dep-graph -Z stable-crate-hash -Z share-generics=yes -C opt-level=0
//@ aux-build: shared_dep.rs
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "main", cfg = "rpass2")]
#![rustc_partition_reused(module = "main", cfg = "rpass3")]

extern crate shared_dep;

fn main() {
    // Use a generic function - with share-generics, this reuses the
    // monomorphization from shared_dep if it exists there
    let x: u32 = shared_dep::generic_fn(42);
    assert_eq!(x, 42);

    // Use another instantiation
    let y: u64 = shared_dep::generic_fn(100);
    assert_eq!(y, 100);

    // Use a generic struct with drop glue (drop glue is also shared)
    let s = shared_dep::GenericBox::new(String::from("hello"));
    assert_eq!(s.get(), "hello");
}
