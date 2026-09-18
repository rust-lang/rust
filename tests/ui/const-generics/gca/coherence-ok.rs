//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(generic_const_items, min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

// computing different values with the same const item should be fine

const ADD1<const N: usize>: usize = N + 1;

trait Trait {}

impl Trait for [(); core::direct_const_arg!(ADD1::<1>)] {}
impl Trait for [(); core::direct_const_arg!(ADD1::<2>)] {}

fn main() {}
