#![feature(transmutability)]
#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

use std::mem::{Assume, TransmuteFrom};

pub fn is_transmutable<const ASSUME_ALIGNMENT: bool>()
where
    (): TransmuteFrom<(), { Assume::SAFETY }>,
{
}

fn foo<const N: usize>() {
    is_transmutable::<{}>();
    //~^ ERROR  the trait bound `(): TransmuteFrom<(), { Assume::SAFETY }>` is not satisfied
    //~| ERROR mismatched types
}

fn main() {}
