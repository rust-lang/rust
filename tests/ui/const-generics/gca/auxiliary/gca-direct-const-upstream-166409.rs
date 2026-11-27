//@ compile-flags: -Znext-solver
#![feature(gca_min_const_items, gca_const_items)]
#![expect(incomplete_features)]

use std::gca;

pub const N: usize = gca!(1);

pub trait Tr {
    const C: usize;
    const D: usize = gca!(5);
}

pub struct S;

impl Tr for S {
    const C: usize = gca!(3);
}
