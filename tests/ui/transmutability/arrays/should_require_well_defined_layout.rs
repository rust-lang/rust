//! An array must have a well-defined layout to participate in a transmutation.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume::ALIGNMENT
                .and(Assume::LIFETIMES)
                .and(Assume::SAFETY)
                .and(Assume::VALIDITY)
        }>
    {}
}

fn should_reject_repr_rust()
{
    fn unit() {
        type repr_rust = [String; 0];
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }

    fn singleton() {
        type repr_rust = [String; 1];
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }

    fn duplex() {
        type repr_rust = [String; 2];
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }
}

fn should_accept_repr_C()
{
    fn unit() {
        #[repr(C)] struct repr_c(u8, u16, u8);
        type array = [repr_c; 0];
        assert::is_maybe_transmutable::<array, ()>();
        assert::is_maybe_transmutable::<i128, array>();
    }

    fn singleton() {
        #[repr(C)] struct repr_c(u8, u16, u8);
        type array = [repr_c; 1];
        assert::is_maybe_transmutable::<array, repr_c>();
        assert::is_maybe_transmutable::<repr_c, array>();
    }

    fn duplex() {
        #[repr(C)] struct repr_c(u8, u16, u8);
        #[repr(C)] struct duplex(repr_c, repr_c);
        type array = [repr_c; 2];
        assert::is_maybe_transmutable::<array, duplex>();
        assert::is_maybe_transmutable::<duplex, array>();
    }
}
