//@ aux-build:system-allocator.rs
//@ aux-build:system-allocator2.rs
//@ no-prefer-dynamic

extern crate system_allocator;
extern crate system_allocator2;

fn main() {}

//~? ERROR the `#[global_allocator]` in system_allocator conflicts with global allocator in: system_allocator2
