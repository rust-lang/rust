//@ check-pass
//! A struct must have a well-defined layout to participate in a transmutation.

#![crate_type = "lib"]
#![feature(transmutability)]
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
    fn unit() {
        struct repr_rust;
        assert::is_maybe_transmutable::<repr_rust, ()>();
        assert::is_maybe_transmutable::<u128, repr_rust>();
    }

    fn tuple() {
        struct repr_rust();
        assert::is_maybe_transmutable::<repr_rust, ()>();
        assert::is_maybe_transmutable::<u128, repr_rust>();
    }

    fn braces() {
        struct repr_rust{}
        assert::is_maybe_transmutable::<repr_rust, ()>();
        assert::is_maybe_transmutable::<u128, repr_rust>();
    }

    fn aligned() {
        #[repr(align(1))] struct repr_rust{}
        assert::is_maybe_transmutable::<repr_rust, ()>();
        assert::is_maybe_transmutable::<u128, repr_rust>();
    }

    fn packed() {
        #[repr(packed)] struct repr_rust{}
        assert::is_maybe_transmutable::<repr_rust, ()>();
        assert::is_maybe_transmutable::<u128, repr_rust>();
    }

    fn nested() {
        struct repr_rust;
        #[repr(C)] struct repr_c(repr_rust);
        assert::is_maybe_transmutable::<repr_c, ()>();
        assert::is_maybe_transmutable::<u128, repr_c>();
    }
}

fn should_accept_repr_c()
{
    fn unit() {
        #[repr(C)] struct repr_c;
        assert::is_maybe_transmutable::<repr_c, ()>();
        assert::is_maybe_transmutable::<i128, repr_c>();
    }

    fn tuple() {
        #[repr(C)] struct repr_c();
        assert::is_maybe_transmutable::<repr_c, ()>();
        assert::is_maybe_transmutable::<i128, repr_c>();
    }

    fn braces() {
        #[repr(C)] struct repr_c{}
        assert::is_maybe_transmutable::<repr_c, ()>();
        assert::is_maybe_transmutable::<i128, repr_c>();
    }
}
