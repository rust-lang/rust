#![feature(transmutability)]

use std::mem::{Assume, TransmuteFrom};

#[repr(C)]
struct W<'a>(&'a ());

fn test<'a>()
where
    W<'a>: TransmuteFrom<
            (),
            { Assume { alignment: true, lifetimes: true, safety: true, validity: true } },
        >,
{
}

fn main() {
    test();
    //~^ ERROR `()` cannot be safely transmuted into `W<'_>`
}
