//@ compile-flags: -Znext-solver
#![feature(generic_const_items, gca_min_const_items, gca_const_items)]
#![expect(incomplete_features)]

use std::gca;

pub const ADD1<const N: usize>: usize = N + 1;

pub struct W<const N: usize>(pub [u8; gca!(ADD1::<N>)]);

pub fn mk<const N: usize>() -> W<N> {
    W([0; gca!(ADD1::<N>)])
}
