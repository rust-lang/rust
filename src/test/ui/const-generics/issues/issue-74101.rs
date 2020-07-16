// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

fn test<const N: [u8; 1 + 2]>() {}

struct Foo<const N: [u8; 1 + 2]>;

fn main() {}
