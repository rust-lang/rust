//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

const ADD1<const N: usize>: usize = N + 1;

const INC<const N: usize>: usize = N + 1;

const ARR: [(); core::direct_const_arg!(ADD1::<0>)] = [(); core::direct_const_arg!(INC::<0>)];

fn main() {}
