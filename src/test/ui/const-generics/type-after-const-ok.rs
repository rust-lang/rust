// run-pass
// Verifies that having generic parameters after constants is permitted

#![feature(const_generics)]
#![allow(incomplete_features)]

#[allow(dead_code)]
struct A<const N: usize, T>(T);

fn main() {}
