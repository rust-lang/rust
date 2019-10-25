// run-pass
// compile-flags: -Z chalk

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub struct Foo<T, const N: usize>([T; N]);
impl<T, const N: usize> Foo<T, {N}> {}

fn main() {}
