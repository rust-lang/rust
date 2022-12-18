#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            Context,
            { Assume { alignment: true, lifetimes: true, safety: true, validity: true } },
        >,
    {
    }
}

fn test() {
    type NaughtyLenArray = [u32; 3.14159]; //~ ERROR mismatched types
    type JustUnit = ();
    assert::is_maybe_transmutable::<JustUnit, NaughtyLenArray>();
}
