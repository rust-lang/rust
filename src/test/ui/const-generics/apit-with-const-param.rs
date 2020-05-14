// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait Trait {}

fn f<const N: usize>(_: impl Trait) {}

fn main() {}
