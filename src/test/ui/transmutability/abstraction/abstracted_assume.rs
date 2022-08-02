// check-pass
//! The implementation should behave correctly when the `ASSUME` parameters are
//! provided indirectly through an abstraction.

#![crate_type = "lib"]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        Context,
        const ASSUME_ALIGNMENT: bool,
        const ASSUME_LIFETIMES: bool,
        const ASSUME_VALIDITY: bool,
        const ASSUME_VISIBILITY: bool,
    >()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            Context,
            ASSUME_ALIGNMENT,
            ASSUME_LIFETIMES,
            ASSUME_VALIDITY,
            ASSUME_VISIBILITY,
        >,
    {}
}

fn direct() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    assert::is_transmutable::<Src, Dst, Context, false, false, false, false>();
}

fn via_const() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    const FALSE: bool = false;

    assert::is_transmutable::<Src, Dst, Context, FALSE, FALSE, FALSE, FALSE>();
}

fn via_associated_const() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    trait Trait {
        const FALSE: bool = true;
    }

    struct Ty;

    impl Trait for Ty {}

    assert::is_transmutable::<
        Src,
        Dst,
        Context,
        {Ty::FALSE},
        {Ty::FALSE},
        {Ty::FALSE},
        {Ty::FALSE}
    >();
}
