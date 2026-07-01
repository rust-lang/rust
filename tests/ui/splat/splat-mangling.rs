//! Test that splat influences symbol mangling.
//@ revisions: default legacy v0
//@ [default] compile-flags: -C opt-level=0
//@ [legacy] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=legacy
//@ [v0] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=v0
//@ [default] build-fail
//@ [legacy] run-fail
//@ [v0] build-fail
//@ incremental

#![allow(incomplete_features)]
#![feature(splat)]

fn main() {
    // Bug #158603 regression test variants
    #[rustfmt::skip]
    let _x: fn(#[splat] (i32,)) = None.unwrap();

    //@ [default] regex-error-pattern: symbol `.*Option.*unwrap.*splat_mangling` is already defined
    //@ [v0] regex-error-pattern: symbol `.*Option.*unwrap.*splat_mangling` is already defined
    //[default,v0]~? ERROR: is already defined
    let x: fn((i32,)) = None.unwrap();
    x((1,));
}
