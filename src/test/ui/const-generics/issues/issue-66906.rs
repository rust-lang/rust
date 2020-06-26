// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

pub struct Tuple;

pub trait Trait<const I: usize> {
    type Input: From<<Self as Trait<I>>::Input>;
}

fn main() {}
