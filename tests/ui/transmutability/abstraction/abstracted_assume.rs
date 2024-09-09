//@ check-pass
//! The implementation should behave correctly when the `ASSUME` parameters are
//! provided indirectly through an abstraction.

#![crate_type = "lib"]
#![feature(adt_const_params)]
#![feature(transmutability)]
#![allow(dead_code, incomplete_features, non_camel_case_types)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<
        Src,
        Dst,
        const ASSUME: std::mem::Assume,
    >()
    where
        Dst: TransmuteFrom<
            Src,
            ASSUME,
        >,
    {}
}

fn direct() {
    assert::is_transmutable::<(), (), { std::mem::Assume::NOTHING }>();
}

fn via_const() {
    const FALSE: bool = false;

    assert::is_transmutable::<(), (), { std::mem::Assume::NOTHING }>();
}

fn via_associated_const() {
    trait Trait {
        const FALSE: bool = true;
    }

    struct Ty;

    impl Trait for Ty {}

    assert::is_transmutable::<
        (),
        (),
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
