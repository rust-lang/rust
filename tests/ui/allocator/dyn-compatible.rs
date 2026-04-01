//@ run-pass

// Check that `Allocator` is dyn-compatible, this allows for polymorphic allocators

#![feature(allocator_api)]

use std::alloc::{Allocator, System};

fn ensure_dyn_compatible(_: &dyn Allocator) {}

fn main() {
    ensure_dyn_compatible(&System);
}
