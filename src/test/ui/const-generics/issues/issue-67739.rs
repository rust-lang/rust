// Regression test for #67739

#![allow(incomplete_features)]
#![feature(const_generics)]

use std::mem;

pub trait Trait {
    type Associated: Sized;

    fn associated_size(&self) -> usize {
        [0u8; mem::size_of::<Self::Associated>()];
        //~^ ERROR: array lengths can't depend on generic parameters
        0
    }
}

fn main() {}
