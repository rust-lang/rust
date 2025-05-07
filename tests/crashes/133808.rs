//@ known-bug: #133808

#![feature(generic_const_exprs, transmutability)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src>,
    {
    }
}

pub fn main() {}
