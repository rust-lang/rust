// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait Trait<const T: ()> {}

fn main() {}
