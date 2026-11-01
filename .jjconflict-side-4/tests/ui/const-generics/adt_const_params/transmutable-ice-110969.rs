// ICE Inconsistent rustc_transmute::is_transmutable(...) result, got Yes
// issue: rust-lang/rust#110969
#![feature(adt_const_params, generic_const_exprs, transmutability)]
#![allow(incomplete_features, unstable_features)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, Dst, Context, const ASSUME: std::mem::Assume>()
    where
        Dst: TransmuteFrom<Src, Context, ASSUME>,
        //~^ ERROR trait takes at most 2 generic arguments but 3 generic arguments were supplied
    {
    }
}

fn via_associated_const() {
    struct Context;
    #[repr(C)]
    struct Src;
    #[repr(C)]
    struct Dst;

    trait Trait {
        const FALSE: bool = assert::is_transmutable::<Src, Dst, Context, {}>();
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
    }
}

pub fn main() {}
