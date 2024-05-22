//@ compile-flags: --crate-type=lib
//@ check-pass
//@ revisions: e2015 e2018
//
//@[e2018] edition:2018

// Verify that gates for the `f16` and `f128` features do not apply to user types

mod binary16 {
    #[allow(non_camel_case_types)]
    pub struct f16(u16);
}

mod binary128 {
    #[allow(non_camel_case_types)]
    pub struct f128(u128);
}

pub use binary128::f128;
pub use binary16::f16;

mod private16 {
    use crate::f16;

    pub trait SealedHalf {}
    impl SealedHalf for f16 {}
}

mod private128 {
    use crate::f128;

    pub trait SealedQuad {}
    impl SealedQuad for f128 {}
}
