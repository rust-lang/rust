//@ check-pass

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(opaque_generic_const_args)]
#![expect(incomplete_features)]

type const ADD1<const N: usize>: usize = const { N + 1 };

type const INC<const N: usize>: usize = ADD1::<N>;

type const ONE: usize = ADD1::<0>;

type const OTHER_ONE: usize = INC::<0>;

const ARR: [(); ADD1::<0>] = [(); INC::<0>];

fn main() {}
