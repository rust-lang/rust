//! Test using `#[rustc_splat]` on tuple arguments of pointers to generic functions.
//@ run-pass

#![expect(incomplete_features)]
#![feature(splat, tuple_trait)]

use std::fmt::Debug;
use std::marker::Tuple;

fn generic<T: Tuple + Debug>(#[rustc_splat] a: T) -> String {
    format!("{a:?}")
}

// FIXME(rustfmt): the attribute gets deleted by rustfmt
#[rustfmt::skip]
fn main() {
    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> String
        = generic as fn(#[rustc_splat] (u32, i8)) -> String;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> String
        = generic::<(u32, i8)> as fn(#[rustc_splat] (u32, i8)) -> String;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    let fn_ptr = generic as fn(#[rustc_splat] (u32, i8)) -> String;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    let fn_ptr = generic::<(u32, i8)> as fn(#[rustc_splat] (u32, i8)) -> String;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> String = generic as _;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> String = generic::<(u32, i8)> as _;
    assert_eq!(fn_ptr(1, -2), "(1, -2)");
    assert_eq!(fn_ptr(1u32, -2i8), "(1, -2)");

    // Now without explicit `as`, this requires turbofish
    let fn_ptr: fn(#[rustc_splat] (f64, i8)) -> String = generic::<(f64, i8)>;
    assert_eq!(fn_ptr(3.5, -2), "(3.5, -2)");
    assert_eq!(fn_ptr(3.5f64, -2i8), "(3.5, -2)");

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    let fn_ptr = generic;
    assert_eq!(fn_ptr(-1, 2, 3.5), "(-1, 2, 3.5)");
    assert_eq!(fn_ptr(-1i8, 2u32, 3.5f64), "(-1, 2, 3.5)");

    #[expect(unused_variables)]
    let fn_ptr = generic::<(i8, u32, f64)>;
    assert_eq!(fn_ptr(-1, 2, 3.5), "(-1, 2, 3.5)");
    assert_eq!(fn_ptr(-1i8, 2u32, 3.5f64), "(-1, 2, 3.5)");
}
