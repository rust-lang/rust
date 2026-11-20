//@ check-pass
//! The presence of an `align(X)` annotation must be accounted for.

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

fn should_pad_explicitly_aligned_field() {
    #[derive(Clone, Copy)] #[repr(u8)] enum V0u8 { V = 0 }
    #[derive(Clone, Copy)] #[repr(u8)] enum V1u8 { V = 1 }

    #[repr(align(1))]
    pub union Uninit {
        a: (),
        b: V1u8,
    }

    #[repr(align(2))]
    pub union align_2 {
        a: V0u8,
    }

    #[repr(C)] struct ImplicitlyPadded(align_2, V0u8);
    #[repr(C)] struct ExplicitlyPadded(V0u8, Uninit, V0u8);

    // An implementation that (incorrectly) does not place a padding byte after
    // `align_2` will, incorrectly, reject the following transmutations.
    assert::is_maybe_transmutable::<ImplicitlyPadded, ExplicitlyPadded>();
    assert::is_maybe_transmutable::<ExplicitlyPadded, ImplicitlyPadded>();
}
