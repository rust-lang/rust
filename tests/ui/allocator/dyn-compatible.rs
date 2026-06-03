//@ run-pass

// Check that `Alloc` is dyn-compatible, this allows for polymorphic allocators

#![feature(allocator_api)]

use std::alloc::{Alloc, Allocator, System};

fn ensure_dyn_compatible(_: &dyn Alloc) {}

fn main() {
    ensure_dyn_compatible(System.alloc_ref());
}
