#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

use std::mem::MaybeUninit;

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::SAFETY }>
    {}
}

fn validity() {
    // An initialized byte is a valid uninitialized byte.
    assert::is_maybe_transmutable::<u8, MaybeUninit<u8>>();

    // An uninitialized byte is never a valid initialized byte.
    assert::is_maybe_transmutable::<MaybeUninit<u8>, u8>(); //~ ERROR: cannot be safely transmuted
}

fn padding() {
    #[repr(align(8))]
    struct Align8;

    #[repr(u8)]
    enum ImplicitlyPadded {
        A(Align8),
    }

    #[repr(u8)]
    enum V0 {
        V0 = 0,
    }

    #[repr(C)]
    struct ExplicitlyPadded(V0, MaybeUninit<[u8; 7]>);

    assert::is_maybe_transmutable::<ExplicitlyPadded, ImplicitlyPadded>();
    assert::is_maybe_transmutable::<ImplicitlyPadded, ExplicitlyPadded>();
}
