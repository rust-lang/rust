//@ check-pass
pub trait Trait<const N: usize>: From<<Self as Trait<N>>::Item> {
  type Item;
}

fn main() {}
