// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct Tuple;

pub trait Trait<const I: usize> {
    type Input: From<<Self as Trait<I>>::Input>;
}

fn main() {}
