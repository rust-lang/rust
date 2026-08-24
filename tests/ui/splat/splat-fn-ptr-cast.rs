//@ run-fail
//! Test never type casts to splatted and non-splatted functions.

#![allow(incomplete_features)]
#![feature(splat)]
#![allow(unused_features)]

fn main() {
    // Bug #158603 regression test variants
    #[rustfmt::skip]
    let x: fn(#[rustc_splat] (f32,)) = None.unwrap();
    x(1.0);

    let x: fn((i32,)) = None.unwrap();
    x((1,));
}
