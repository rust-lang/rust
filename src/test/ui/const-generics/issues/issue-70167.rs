// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub trait Trait<const N: usize>: From<<Self as Trait<N>>::Item> {
  type Item;
}

fn main() {}
