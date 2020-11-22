#![feature(const_generics)]
#![allow(incomplete_features)]
struct Foo where [u8; { let _: &'a (); 3 }]: Sized;

fn main() {}
