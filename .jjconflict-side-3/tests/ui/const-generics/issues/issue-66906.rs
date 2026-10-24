//@ check-pass

pub struct Tuple;

pub trait Trait<const I: usize> {
    type Input: From<<Self as Trait<I>>::Input>;
}

fn main() {}
