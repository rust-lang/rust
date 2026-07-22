//! Test that splat influences symbol mangling, regression test for #158644
//@ revisions: default legacy v0
//@ [default] compile-flags: -C opt-level=0
//@ [legacy] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=legacy
//@ [v0] compile-flags: -C opt-level=0 -Z unstable-options -Csymbol-mangling-version=v0
//@ run-fail
//@ incremental

#![allow(incomplete_features)]
#![feature(splat)]

fn main() {
    #[rustfmt::skip]
    let _x: fn(#[splat] (i32,)) = None.unwrap();

    let x: fn((i32,)) = None.unwrap();
    x((1,));
}
