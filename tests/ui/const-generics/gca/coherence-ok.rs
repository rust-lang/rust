//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(generic_const_items, min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

// computing different values with the same const item should be fine

const ADD1<const N: usize>: usize = N + 1;

trait Trait {}

impl Trait for [(); gca!(ADD1::<1>)] {}
impl Trait for [(); gca!(ADD1::<2>)] {}

fn main() {}
