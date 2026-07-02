//! Test that splat is used in symbol mangling.
//@ build-fail
//@ incremental
//@ compile-flags: -C opt-level=0

#![allow(incomplete_features)]
#![feature(splat)]

fn main() {
    // Bug #158603 regression test variants
    #[rustfmt::skip]
    let _x: fn(#[splat] (i32,)) = None.unwrap();

    //@ regex-error-pattern: symbol `.*Option.*unwrap.*splat_mangling` is already defined
    //~? ERROR: is already defined
    let x: fn((i32,)) = None.unwrap();
    x((1,));
}
