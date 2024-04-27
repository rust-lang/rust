#![feature(transmutability)]

use std::mem::{Assume, BikeshedIntrinsicFrom};

#[repr(C)]
struct W<'a>(&'a ());

fn test<'a>()
where
    W<'a>: BikeshedIntrinsicFrom<
            (),
            { Assume { alignment: true, lifetimes: true, safety: true, validity: true } },
        >,
{
}

fn main() {
    test();
    //~^ ERROR `()` cannot be safely transmuted into `W<'_>`
}
