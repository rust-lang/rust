//@ aux-build:system-allocator.rs
//@ no-prefer-dynamic

extern crate system_allocator;

use std::alloc::System;

#[global_allocator]
static A: System = System;

fn main() {}

//~? ERROR the `#[global_allocator]` in this crate conflicts with global allocator in: system_allocator
