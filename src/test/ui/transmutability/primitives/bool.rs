#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]
#![allow(incomplete_features)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, false, false, false, true>
    {}

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, false, false, true, true>
    {}
}

fn contrast_with_u8() {
    assert::is_transmutable::<u8, bool>(); //~ ERROR cannot be safely transmuted
    assert::is_maybe_transmutable::<u8, bool>();
    assert::is_transmutable::<bool, u8>();
}
