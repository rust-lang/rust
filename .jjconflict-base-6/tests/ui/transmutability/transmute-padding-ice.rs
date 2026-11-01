//@ check-pass
//! This UI test was introduced as check-fail by a buggy bug-fix for an ICE. In
//! fact, this transmutation should be valid.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code)]

use std::mem::size_of;

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<
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

    assert_eq!(size_of::<A>(), size_of::<B>());

    assert::is_maybe_transmutable::<B, A>();
}
