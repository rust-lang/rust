//! The implementation must behave well if const values of wrong types are
//! provided.

#![crate_type = "lib"]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<
        Src,
        Dst,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_SAFETY: bool,
        const ASSUME_VALIDITY: bool,
    >()
    where
        Dst: TransmuteFrom<
            Src,
            { from_options(ASSUME_ALIGNMENT, ASSUME_LIFETIMES, ASSUME_SAFETY, ASSUME_VALIDITY) }
        >,
    {}

    const fn from_options(
        alignment: bool,
        lifetimes: bool,
        safety: bool,
        validity: bool,
    ) -> Assume {
        Assume {
            alignment,
            lifetimes,
            safety,
            validity,
        }
    }
}

fn test() {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;
    assert::is_transmutable::<Src, Dst, {0u8}, false, false, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, false, {0u8}, false, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, false, false, {0u8}, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, false, false, false, {0u8}>(); //~ ERROR mismatched types
}
