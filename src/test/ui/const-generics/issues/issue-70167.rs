// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

pub trait Trait<const N: usize>: From<<Self as Trait<N>>::Item> {
  type Item;
}

fn main() {}
