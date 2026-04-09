//@ check-pass
#![feature(generic_const_items, min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

// computing different values with the same type const item should be fine

type const ADD1<const N: usize>: usize = const { N + 1 };

trait Trait {}

impl Trait for [(); ADD1::<1>] {}
impl Trait for [(); ADD1::<2>] {}

fn main() {}
