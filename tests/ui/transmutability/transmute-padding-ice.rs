#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            { Assume { alignment: true, lifetimes: true, safety: true, validity: true } },
        >,
    {
    }
}

fn test() {
    #[repr(C, align(2))]
    struct A(u8, u8);

    #[repr(C)]
    struct B(u8, u8);

    assert::is_maybe_transmutable::<B, A>();
    //~^ ERROR cannot be safely transmuted
}
