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

fn should_pad_explicitly_packed_field() {
    #[derive(Clone, Copy)] #[repr(u8)] enum V0u8 { V = 0 }
    #[derive(Clone, Copy)] #[repr(u8)] enum V1u8 { V = 1 }
    #[derive(Clone, Copy)] #[repr(u8)] enum V2u8 { V = 2 }
    #[derive(Clone, Copy)] #[repr(u32)] enum V3u32 { V = 3 }

    pub union Uninit {
        a: (),
        b: V1u8,
    }

    #[repr(C, packed(2))]
    pub union Packed {
        a: [V3u32; 0],
        b: V0u8,
    }

    #[repr(C)] struct ImplicitlyPadded(Packed, V2u8);
    #[repr(C)] struct ExplicitlyPadded(V0u8, Uninit, V2u8);

    assert::is_maybe_transmutable::<ImplicitlyPadded, ExplicitlyPadded>();
    assert::is_maybe_transmutable::<ExplicitlyPadded, ImplicitlyPadded>();
}
