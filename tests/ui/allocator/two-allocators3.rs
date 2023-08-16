//@aux-build:system-allocator.rs
//@aux-build:system-allocator2.rs
// no-prefer-dynamic
//@error-in-other-file: the `#[global_allocator]` in


extern crate system_allocator;
extern crate system_allocator2;

fn main() {}
