//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
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
    assert::is_transmutable::<u8, bool, { std::mem::Assume::VALIDITY }>();
    assert::is_transmutable::<u8, bool, { std::mem::Assume::NOTHING }>();
    //~^ ERROR cannot be safely transmuted
}

fn via_const() {
    const FALSE: bool = false;
    const TRUE: bool = true;

    assert::is_transmutable::<
        u8,
        bool,
        {
            std::mem::Assume {
                alignment: FALSE,
                lifetimes: FALSE,
                safety: FALSE,
                validity: TRUE,
            }
        },
    >();
    assert::is_transmutable::<
        u8,
        bool, //~ ERROR cannot be safely transmuted
        {
            std::mem::Assume {
                alignment: FALSE,
                lifetimes: FALSE,
                safety: FALSE,
                validity: FALSE,
            }
        },
    >();
}

fn via_associated_const() {
    trait Trait {
        const FALSE: bool = false;
        const TRUE: bool = true;
    }

    struct Ty;

    impl Trait for Ty {}

    assert::is_transmutable::<
        u8,
        bool,
        {
            std::mem::Assume {
                alignment: Ty::FALSE,
                lifetimes: Ty::FALSE,
                safety: Ty::FALSE,
                validity: Ty::TRUE,
            }
        },
    >();
    assert::is_transmutable::<
        u8,
        bool, //~ ERROR cannot be safely transmuted
        {
            std::mem::Assume {
                alignment: Ty::FALSE,
                lifetimes: Ty::FALSE,
                safety: Ty::FALSE,
                validity: Ty::FALSE,
            }
        },
    >();
}
