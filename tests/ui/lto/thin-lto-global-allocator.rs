//@ run-pass
//@ compile-flags: -Clto=thin -C codegen-units=2

#[global_allocator]
static A: std::alloc::System = std::alloc::System;

fn main() {}
