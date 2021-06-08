// check-pass
#![feature(const_evaluatable_checked, const_generics)]
#![allow(incomplete_features)]

struct Foo<const N: u8>([u8; N as usize])
where
    [(); N as usize]:;

fn main() {}
