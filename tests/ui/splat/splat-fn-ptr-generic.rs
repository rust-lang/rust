//! Test using `#[splat]` on tuple arguments of pointers to generic functions.
//@ run-pass
//@ check-run-results

#![expect(incomplete_features)]
#![feature(splat, tuple_trait)]

use std::fmt::Debug;
use std::marker::Tuple;

fn generic<T: Tuple + Debug>(#[splat] a: T) {
    println!("generic: {a:?}");
}

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = generic as fn(#[splat] (u32, i8));
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = generic::<(u32, i8)> as fn(#[splat] (u32, i8));
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    #[rustfmt::skip]
    let fn_ptr = generic as fn(#[splat] (u32, i8));
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    #[rustfmt::skip]
    let fn_ptr = generic::<(u32, i8)> as fn(#[splat] (u32, i8));
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = generic as _;
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (u32, i8)) = generic::<(u32, i8)> as _;
    fn_ptr(1, -2);
    fn_ptr(1u32, -2i8);

    // Now without explicit `as`, this requires turbofish
    #[rustfmt::skip]
    let fn_ptr: fn(#[splat] (f64, i8)) = generic::<(f64, i8)>;
    fn_ptr(3.5, -2);
    fn_ptr(3.5f64, -2i8);

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_ptr = generic;
    fn_ptr(-1, 2, 3.5);
    fn_ptr(-1i8, 2u32, 3.5f64);

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_ptr = generic::<(i8, u32, f64)>;
    fn_ptr(-1, 2, 3.5);
    fn_ptr(-1i8, 2u32, 3.5f64);
}
