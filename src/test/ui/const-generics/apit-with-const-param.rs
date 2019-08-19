// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

trait Trait {}

fn f<const N: usize>(_: impl Trait) {}

fn main() {}
