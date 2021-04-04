// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub trait Trait<const N: usize>: From<<Self as Trait<N>>::Item> {
  type Item;
}

fn main() {}
