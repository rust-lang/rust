// Test for #134978, which causes an deadlock bug
//
//@ compile-flags: -Z threads=12

#![feature(generic_const_exprs)]

pub struct Struct<const N: usize>;

impl<const N: usize> Struct<N> {
    pub const OK: usize = 0;
}

fn main() {
    function::<0>();
}

fn function<const NUM_CARDS: usize>()
where
    [(); Struct::<{ NUM_CARDS + 0 }>::OK]:,
{
}
