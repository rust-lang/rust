//@ check-pass

#![crate_type = "lib"]
#![feature(transmutability, transparent_unions)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: true,
                lifetimes: true,
                safety: true,
                validity: true,
            }
        }>
    {}
}

fn should_accept_repr_rust()
{
    union repr_rust {
        a: u8
    }

    assert::is_maybe_transmutable::<repr_rust, ()>();
    assert::is_maybe_transmutable::<u128, repr_rust>();
}

fn should_accept_repr_c()
{
    #[repr(C)]
    union repr_c {
        a: u8
    }

    struct repr_rust;
    assert::is_maybe_transmutable::<repr_c, ()>();
    assert::is_maybe_transmutable::<u128, repr_c>();
}


fn should_accept_transparent()
{
    #[repr(transparent)]
    union repr_transparent {
        a: u8
    }

    assert::is_maybe_transmutable::<repr_transparent, ()>();
    assert::is_maybe_transmutable::<u128, repr_transparent>();
}
