//! Test using `#[rustc_splat]` on tuple arguments of pointers to pointers to simple functions.
//! Bug #158603 regression test
//@ run-pass

#![expect(incomplete_features)]
#![feature(splat)]

use std::ptr;

fn tuple_args(#[rustc_splat] (a, b): (u32, i8)) -> (i8, u32) {
    // Permute the returned values as a codegen test
    (b, a)
}

fn splat_non_terminal_arg(#[rustc_splat] (a, b): (u32, i8), c: f64) -> (i8, f64, u32) {
    // Permute the returned values as a codegen test
    (b, c, a)
}

// FIXME(rustfmt): the attribute gets deleted by rustfmt
#[rustfmt::skip]
fn main() {
    let fn_pp: &fn(#[rustc_splat] (u32, i8)) -> (i8, u32)
        = &(tuple_args as fn(#[rustc_splat] (u32, i8)) -> (i8, u32));
    assert_eq!((*fn_pp)(1, 2), (2, 1));
    assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));

    let fn_pp: &fn(#[rustc_splat] (u32, i8)) -> (i8, u32) = &(tuple_args as _);
    assert_eq!((*fn_pp)(1, 2), (2, 1));
    assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));

    let fn_pp = &(tuple_args as fn(#[rustc_splat] (u32, i8)) -> (i8, u32));
    assert_eq!((*fn_pp)(1, 2), (2, 1));
    assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    let fn_pp = &tuple_args;
    assert_eq!((*fn_pp)(1, 2), (2, 1));
    assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));

    // Now with *const
    let fn_pp: *const fn(#[rustc_splat] (u32, i8)) -> (i8, u32)
        = ptr::from_ref(&(tuple_args as fn(#[rustc_splat] (u32, i8)) -> (i8, u32)));
    unsafe {
        assert_eq!((*fn_pp)(1, 2), (2, 1));
        assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));
    }

    let fn_pp: *const fn(#[rustc_splat] (u32, i8)) -> (i8, u32) = ptr::from_ref(&(tuple_args as _));
    unsafe {
        assert_eq!((*fn_pp)(1, 2), (2, 1));
        assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));
    }

    let fn_pp = ptr::from_ref(&(tuple_args as fn(#[rustc_splat] (u32, i8)) -> (i8, u32)));
    unsafe {
        assert_eq!((*fn_pp)(1, 2), (2, 1));
        assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));
    }

    #[expect(unused_variables)]
    let fn_pp = ptr::from_ref(&tuple_args);
    // FIXME(unsafe): dereferencing *const should require unsafe
    assert_eq!((*fn_pp)(1, 2), (2, 1));
    assert_eq!((*fn_pp)(1u32, 2i8), (2i8, 1u32));

    // Now with *mut and non-terminal splat
    let fn_pp: *mut fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32)
        = ptr::from_mut(
            &mut (splat_non_terminal_arg as fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32))
        );
    unsafe {
        assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
        assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));
    }

    let fn_pp: *mut fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32)
        = ptr::from_mut(&mut (splat_non_terminal_arg as _));
    unsafe {
        assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
        assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));
    }

    let fn_pp = ptr::from_mut(
            &mut (splat_non_terminal_arg as fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32))
        );
    unsafe {
        assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
        assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));
    }

    #[expect(unused_variables)]
    let fn_pp = ptr::from_mut(&mut splat_non_terminal_arg);
    // FIXME(unsafe): dereferencing *mut should require unsafe
    assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
    assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));

    // Now with & as *const and non-terminal splat
    let fn_pp: *const fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32)
        = &(splat_non_terminal_arg as fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32));
    unsafe {
        assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
        assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));
    }

    let fn_pp: *const fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32)
        = &(splat_non_terminal_arg as _);
    unsafe {
        assert_eq!((*fn_pp)(1, 2, 3.5), (2, 3.5, 1));
        assert_eq!((*fn_pp)(1u32, 2i8, 3.5f64), (2i8, 3.5f64, 1u32));
    }
}
