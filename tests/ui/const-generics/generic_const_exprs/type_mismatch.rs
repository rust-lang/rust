#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Q {
    const ASSOC: usize;
}

impl<const N: u64> Q for [u8; N] {}
//~^ ERROR not all trait items implemented
//~| ERROR the constant `N` is not of type `usize`

pub fn q_user() -> [u8; <[u8; 13] as Q>::ASSOC] {}
//~^ ERROR the constant `13` is not of type `u64`
//~| ERROR mismatched types

pub fn main() {}
