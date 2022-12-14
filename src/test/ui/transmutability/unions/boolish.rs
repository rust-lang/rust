// check-pass

#![crate_type = "lib"]
#![feature(transmutability)]
#![feature(marker_trait_attr)]
#![allow(dead_code)]
#![allow(incomplete_features)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, { Assume::SAFETY }>
    {}
}

fn should_match_bool() {
    #[derive(Copy, Clone)] #[repr(u8)] pub enum False { V = 0 }
    #[derive(Copy, Clone)] #[repr(u8)] pub enum True { V = 1 }

    #[repr(C)]
    pub union Bool {
        pub f: False,
        pub t: True,
    }

    assert::is_transmutable::<Bool, bool>();
    assert::is_transmutable::<bool, Bool>();
}
