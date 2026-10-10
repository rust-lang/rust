//! Test using `#[rustc_splat]` on tuple arguments of pointers to simple functions.
//! Bug #158603 regression test
//@ run-pass

#![expect(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[rustc_splat] (a, b): (u32, i8)) -> (u32, i8) {
    (a, b)
}

fn splat_non_terminal_arg(#[rustc_splat] (a, b): (u32, i8), c: f64) -> (f64, i8, u32) {
    // Permute the returned values as a codegen test
    (c, b, a)
}

// FIXME(rustfmt): the attribute gets deleted by rustfmt
#[rustfmt::skip]
fn main() {
    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> (u32, i8)
        = tuple_args as fn(#[rustc_splat] (u32, i8)) -> (u32, i8);
    assert_eq!(fn_ptr(1, 2), (1, 2));
    assert_eq!(fn_ptr(1u32, 2i8), (1u32, 2i8));

    let fn_ptr = tuple_args as fn(#[rustc_splat] (u32, i8)) -> (u32, i8);
    assert_eq!(fn_ptr(1, 2), (1, 2));
    assert_eq!(fn_ptr(1u32, 2i8), (1u32, 2i8));

    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> (u32, i8) = tuple_args as _;
    assert_eq!(fn_ptr(1, 2), (1, 2));
    assert_eq!(fn_ptr(1u32, 2i8), (1u32, 2i8));

    // Now without explicit `as`
    let fn_ptr: fn(#[rustc_splat] (u32, i8), f64) -> (f64, i8, u32) = splat_non_terminal_arg;
    assert_eq!(fn_ptr(1, 2, 3.5), (3.5,  2, 1));
    assert_eq!(fn_ptr(1u32, 2i8, 3.5f64), (3.5f64, 2i8, 1u32));

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    let fn_ptr = splat_non_terminal_arg;
    assert_eq!(fn_ptr(1, 2, 3.5), (3.5, 2, 1));
    assert_eq!(fn_ptr(1u32, 2i8, 3.5f64), (3.5f64, 2i8, 1u32));
}
