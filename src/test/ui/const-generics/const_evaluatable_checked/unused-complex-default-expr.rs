// check-pass
#![feature(const_generics, const_evaluatable_checked, const_generics_defaults)]
#![allow(incomplete_features)]
struct Foo<const N: usize, const M: usize = { N + 1 }>;
struct Bar<const N: usize>(Foo<N, 3>);
fn main() {}
