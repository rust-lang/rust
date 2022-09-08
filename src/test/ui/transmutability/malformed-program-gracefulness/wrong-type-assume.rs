//! The implementation must behave well if const values of wrong types are
//! provided.

#![crate_type = "lib"]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<
        Src,
        Dst,
        Context,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_SAFETY: bool,
        const ASSUME_VALIDITY: bool,
    >()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            Context,
            { from_options(ASSUME_ALIGNMENT, ASSUME_LIFETIMES, ASSUME_SAFETY, ASSUME_VALIDITY) }
            //~^ ERROR E0080
            //~| ERROR E0080
            //~| ERROR E0080
            //~| ERROR E0080
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
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;
    assert::is_transmutable::<Src, Dst, Context, {0u8}, false, false, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, Context, false, {0u8}, false, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, Context, false, false, {0u8}, false>(); //~ ERROR mismatched types
    assert::is_transmutable::<Src, Dst, Context, false, false, false, {0u8}>(); //~ ERROR mismatched types
}
