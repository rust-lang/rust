#![allow(dead_code)]

#[repr(i128)] //~ ERROR: attribute should be applied to an enum
struct Foo;

#[repr(u128)] //~ ERROR: attribute should be applied to an enum
struct Bar;

fn main() {}
