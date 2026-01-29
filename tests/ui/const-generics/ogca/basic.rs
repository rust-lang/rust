//@ check-pass

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(opaque_generic_const_args)]
#![expect(incomplete_features)]

#[type_const]
const ADD1<const N: usize>: usize = const { N + 1 };

#[type_const]
const INC<const N: usize>: usize = ADD1::<N>;

#[type_const]
const ONE: usize = ADD1::<0>;

#[type_const]
const OTHER_ONE: usize = INC::<0>;

const ARR: [(); ADD1::<0>] = [(); INC::<0>];

fn main() {}
