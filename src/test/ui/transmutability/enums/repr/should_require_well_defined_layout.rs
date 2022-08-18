//! An enum must have a well-defined layout to participate in a transmutation.

#![crate_type = "lib"]
#![feature(repr128)]
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

fn should_reject_repr_rust() {
    fn void() {
        enum repr_rust {}
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }

    fn singleton() {
        enum repr_rust { V }
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }

    fn duplex() {
        enum repr_rust { A, B }
        assert::is_maybe_transmutable::<repr_rust, ()>(); //~ ERROR cannot be safely transmuted
        assert::is_maybe_transmutable::<u128, repr_rust>(); //~ ERROR cannot be safely transmuted
    }
}

fn should_accept_primitive_reprs()
{
    fn should_accept_repr_i8() {
        #[repr(i8)] enum repr_i8 { V }
        assert::is_maybe_transmutable::<repr_i8, ()>();
        assert::is_maybe_transmutable::<i8, repr_i8>();
    }

    fn should_accept_repr_u8() {
        #[repr(u8)] enum repr_u8 { V }
        assert::is_maybe_transmutable::<repr_u8, ()>();
        assert::is_maybe_transmutable::<u8, repr_u8>();
    }

    fn should_accept_repr_i16() {
        #[repr(i16)] enum repr_i16 { V }
        assert::is_maybe_transmutable::<repr_i16, ()>();
        assert::is_maybe_transmutable::<i16, repr_i16>();
    }

    fn should_accept_repr_u16() {
        #[repr(u16)] enum repr_u16 { V }
        assert::is_maybe_transmutable::<repr_u16, ()>();
        assert::is_maybe_transmutable::<u16, repr_u16>();
    }

    fn should_accept_repr_i32() {
        #[repr(i32)] enum repr_i32 { V }
        assert::is_maybe_transmutable::<repr_i32, ()>();
        assert::is_maybe_transmutable::<i32, repr_i32>();
    }

    fn should_accept_repr_u32() {
        #[repr(u32)] enum repr_u32 { V }
        assert::is_maybe_transmutable::<repr_u32, ()>();
        assert::is_maybe_transmutable::<u32, repr_u32>();
    }

    fn should_accept_repr_i64() {
        #[repr(i64)] enum repr_i64 { V }
        assert::is_maybe_transmutable::<repr_i64, ()>();
        assert::is_maybe_transmutable::<i64, repr_i64>();
    }

    fn should_accept_repr_u64() {
        #[repr(u64)] enum repr_u64 { V }
        assert::is_maybe_transmutable::<repr_u64, ()>();
        assert::is_maybe_transmutable::<u64, repr_u64>();
    }

    fn should_accept_repr_i128() {
        #[repr(i128)] enum repr_i128 { V }
        assert::is_maybe_transmutable::<repr_i128, ()>();
        assert::is_maybe_transmutable::<i128, repr_i128>();
    }

    fn should_accept_repr_u128() {
        #[repr(u128)] enum repr_u128 { V }
        assert::is_maybe_transmutable::<repr_u128, ()>();
        assert::is_maybe_transmutable::<u128, repr_u128>();
    }

    fn should_accept_repr_isize() {
        #[repr(isize)] enum repr_isize { V }
        assert::is_maybe_transmutable::<repr_isize, ()>();
        assert::is_maybe_transmutable::<isize, repr_isize>();
    }

    fn should_accept_repr_usize() {
        #[repr(usize)] enum repr_usize { V }
        assert::is_maybe_transmutable::<repr_usize, ()>();
        assert::is_maybe_transmutable::<usize, repr_usize>();
    }
}

fn should_accept_repr_C() {
    #[repr(C)] enum repr_c { V }
    assert::is_maybe_transmutable::<repr_c, ()>();
    assert::is_maybe_transmutable::<i128, repr_c>();
}
