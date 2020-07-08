#![allow(dead_code)]

#[repr(i128)] //~ ERROR: attribute should be applied to enum
struct Foo;

#[repr(u128)] //~ ERROR: attribute should be applied to enum
struct Bar;

fn main() {}
