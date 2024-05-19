//@ run-pass

// Check that `Allocator` is object safe, this allows for polymorphic allocators

#![feature(allocator_api)]

use std::alloc::{Allocator, System};

fn ensure_object_safe(_: &dyn Allocator) {}

fn main() {
    ensure_object_safe(&System);
}
