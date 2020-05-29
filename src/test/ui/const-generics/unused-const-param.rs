// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct A<const N: usize>; // ok

fn main() {}
