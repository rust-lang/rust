//! Transmutations involving references are not yet supported.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume {
                alignment: true,
                lifetimes: true,
                safety: true,
                validity: true,
            }
        }>
    {}
}

fn not_yet_implemented() {
    #[repr(C)] struct Unit;
    assert::is_maybe_transmutable::<&'static Unit, &'static Unit>(); //~ ERROR cannot be safely transmuted
}
