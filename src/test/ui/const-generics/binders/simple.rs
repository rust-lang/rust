// check-pass
#![feature(const_generics)]
#![allow(incomplete_features)]
struct Foo<T>(T) where for<'a> [T; { let _: &'a (); 3 }]: Sized;

fn main() {}
