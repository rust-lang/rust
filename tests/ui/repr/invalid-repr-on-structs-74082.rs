// https://github.com/rust-lang/rust/issues/74082
#![allow(dead_code)]

#[repr(i128)] //~ ERROR: attribute cannot be used on
struct Foo;

#[repr(u128)] //~ ERROR: attribute cannot be used on
struct Bar;

fn main() {}
