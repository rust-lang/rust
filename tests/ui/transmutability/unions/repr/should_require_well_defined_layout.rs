//! A struct must have a well-defined layout to participate in a transmutation.

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

fn should_reject_repr_rust()
{
    union repr_rust {
        a: u8
    }

    assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
    assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
}

fn should_accept_repr_C()
{
    #[repr(C)]
    union repr_c {
        a: u8
    }

    struct repr_rust;
    assert::is_maybe_transmutable::<repr_c, ()>();
    assert::is_maybe_transmutable::<u128, repr_c>();
}
