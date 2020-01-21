// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Foo<const A: usize, const B: usize>;

impl<const A: usize> Foo<1, A> {} // ok

fn main() {}
