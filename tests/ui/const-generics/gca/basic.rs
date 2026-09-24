//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

const ADD1<const N: usize>: usize = N + 1;

const INC<const N: usize>: usize = gca!(ADD1::<N>);

const ARR: [(); gca!(ADD1::<0>)] = [(); gca!(INC::<0>)];

fn main() {}
