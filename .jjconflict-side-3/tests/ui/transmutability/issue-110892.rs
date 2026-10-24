//@ check-fail
#![feature(generic_const_exprs, transmutability)]
#![allow(incomplete_features)]

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

    // This should not cause an ICE
    const fn from_options(
        , //~ ERROR expected parameter name, found `,`
        , //~ ERROR expected parameter name, found `,`
        , //~ ERROR expected parameter name, found `,`
        , //~ ERROR expected parameter name, found `,`
    ) -> Assume {} //~ ERROR mismatched types
}

fn main() {
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    assert::is_transmutable::<Src, Dst, false, false, { true }, false>();
}
