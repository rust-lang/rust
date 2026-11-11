//@ check-pass
//! Adapted from https://rust-lang.github.io/unsafe-code-guidelines/layout/enums.html#explicit-repr-annotation-without-c-compatibility
#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

use std::mem::MaybeUninit;

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, {
            Assume {
                alignment: false,
                lifetimes: false,
                safety: true,
                validity: true,
            }
        }>
    {}
}

#[repr(u8)] enum V0 { V = 0 }
#[repr(u8)] enum V1 { V = 1 }

fn repr_u8() {
    #[repr(u8)]
    enum TwoCases {
        A(u8, u16),     // 0x00 INIT INIT INIT
        B(u16),         // 0x01 PADD INIT INIT
    }

    const _: () = {
        assert!(std::mem::size_of::<TwoCases>() == 4);
    };

    #[repr(C)] struct TwoCasesA(V0, u8, u8, u8);
    #[repr(C)] struct TwoCasesB(V1, MaybeUninit<u8>, u8, u8);

    assert::is_transmutable::<TwoCasesA, TwoCases>();
    assert::is_transmutable::<TwoCasesB, TwoCases>();

    assert::is_maybe_transmutable::<TwoCases, TwoCasesA>();
    assert::is_maybe_transmutable::<TwoCases, TwoCasesB>();
}

fn repr_c_u8() {
    #[repr(C, u8)]
    enum TwoCases {
        A(u8, u16),     // 0x00 PADD INIT PADD INIT INIT
        B(u16),         // 0x01 PADD INIT INIT PADD PADD
    }

    const _: () = {
        assert!(std::mem::size_of::<TwoCases>() == 6);
    };

    #[repr(C)] struct TwoCasesA(V0, MaybeUninit<u8>, u8, MaybeUninit<u8>, u8, u8);
    #[repr(C)] struct TwoCasesB(V1, MaybeUninit<u8>, u8, u8, MaybeUninit<u8>, MaybeUninit<u8>);

    assert::is_transmutable::<TwoCasesA, TwoCases>();
    assert::is_transmutable::<TwoCasesB, TwoCases>();

    assert::is_maybe_transmutable::<TwoCases, TwoCasesA>();
    assert::is_maybe_transmutable::<TwoCases, TwoCasesB>();
}
