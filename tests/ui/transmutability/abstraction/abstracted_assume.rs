// check-pass
//! The implementation should behave correctly when the `ASSUME` parameters are
//! provided indirectly through an abstraction.

#![crate_type = "lib"]
#![feature(adt_const_params)]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        Context,
        const ASSUME: std::mem::Assume,
    >()
    where
        Dst: BikeshedIntrinsicFrom<
            Src,
            Context,
            ASSUME,
        >,
    {}
}

fn direct() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    assert::is_transmutable::<Src, Dst, Context, { std::mem::Assume::NOTHING }>();
}

fn via_const() {
    struct Context;
    #[repr(C)] struct Src;
    #[repr(C)] struct Dst;

    const FALSE: bool = false;

    assert::is_transmutable::<Src, Dst, Context, { std::mem::Assume::NOTHING }>();
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
        {
            std::mem::Assume {
                alignment: {Ty::FALSE},
                lifetimes: {Ty::FALSE},
                safety: {Ty::FALSE},
                validity: {Ty::FALSE},
            }
        }
    >();
}
