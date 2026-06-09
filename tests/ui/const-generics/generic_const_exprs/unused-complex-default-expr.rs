//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
struct Foo<const N: usize, const M: usize = { N + 1 }>;
struct Bar<const N: usize>(Foo<N, 3>);
fn main() {}
