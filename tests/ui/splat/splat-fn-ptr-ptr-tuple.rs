//! Test using `#[splat]` on tuple arguments of pointers to pointers to simple functions.
//! Bug #158603 regression test
//@ run-pass
//@ check-run-results

#![expect(incomplete_features)]
#![feature(splat)]

use std::ptr;

fn tuple_args(#[splat] (a, b): (u32, i8)) {
    println!("tuple_args: {a} {b}");
}

fn splat_non_terminal_arg(#[splat] (a, b): (u32, i8), c: f64) {
    println!("splat_non_terminal_arg: {a} {b} {c}");
}

fn main() {
    // FIXME(rustfmt): the attribute gets deleted by rustfmt
    #[rustfmt::skip]
    let fn_pp: &fn(#[splat] (u32, i8)) = &(tuple_args as fn(#[splat] (u32, i8)));
    (*fn_pp)(1, 2);
    (*fn_pp)(1u32, 2i8);

    #[rustfmt::skip]
    let fn_pp: &fn(#[splat] (u32, i8)) = &(tuple_args as _);
    (*fn_pp)(1, 2);
    (*fn_pp)(1u32, 2i8);

    #[rustfmt::skip]
    let fn_pp = &(tuple_args as fn(#[splat] (u32, i8)));
    (*fn_pp)(1, 2);
    (*fn_pp)(1u32, 2i8);

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_pp = &tuple_args;
    (*fn_pp)(1, 2);
    (*fn_pp)(1u32, 2i8);

    // Now with *const
    #[rustfmt::skip]
    let fn_pp: *const fn(#[splat] (u32, i8))
        = ptr::from_ref(&(tuple_args as fn(#[splat] (u32, i8))));
    unsafe {
        (*fn_pp)(1, 2);
        (*fn_pp)(1u32, 2i8);
    }

    #[rustfmt::skip]
    let fn_pp: *const fn(#[splat] (u32, i8)) = ptr::from_ref(&(tuple_args as _));
    unsafe {
        (*fn_pp)(1, 2);
        (*fn_pp)(1u32, 2i8);
    }

    #[rustfmt::skip]
    let fn_pp = ptr::from_ref(&(tuple_args as fn(#[splat] (u32, i8))));
    unsafe {
        (*fn_pp)(1, 2);
        (*fn_pp)(1u32, 2i8);
    }

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_pp = ptr::from_ref(&tuple_args);
    // FIXME(unsafe): dereferencing *const should require unsafe
    (*fn_pp)(1, 2);
    (*fn_pp)(1u32, 2i8);

    // Now with *mut and non-terminal splat
    #[rustfmt::skip]
    let fn_pp: *mut fn(#[splat] (u32, i8), f64)
        = ptr::from_mut(&mut (splat_non_terminal_arg as fn(#[splat] (u32, i8), f64)));
    unsafe {
        (*fn_pp)(1, 2, 3.5);
        (*fn_pp)(1u32, 2i8, 3.5f64);
    }

    #[rustfmt::skip]
    let fn_pp: *mut fn(#[splat] (u32, i8), f64) = ptr::from_mut(&mut (splat_non_terminal_arg as _));
    unsafe {
        (*fn_pp)(1, 2, 3.5);
        (*fn_pp)(1u32, 2i8, 3.5f64);
    }

    #[rustfmt::skip]
    let fn_pp = ptr::from_mut(&mut (splat_non_terminal_arg as fn(#[splat] (u32, i8), f64)));
    unsafe {
        (*fn_pp)(1, 2, 3.5);
        (*fn_pp)(1u32, 2i8, 3.5f64);
    }

    // FIXME(unused_variables): This is obviously used
    #[expect(unused_variables)]
    #[rustfmt::skip]
    let fn_pp = ptr::from_mut(&mut splat_non_terminal_arg);
    // FIXME(unsafe): dereferencing *mut should require unsafe
    (*fn_pp)(1, 2, 3.5);
    (*fn_pp)(1u32, 2i8, 3.5f64);

    // Now with & as *const and non-terminal splat
    #[rustfmt::skip]
    let fn_pp: *const fn(#[splat] (u32, i8), f64)
        = &(splat_non_terminal_arg as fn(#[splat] (u32, i8), f64));
    unsafe {
        (*fn_pp)(1, 2, 3.5);
        (*fn_pp)(1u32, 2i8, 3.5f64);
    }

    #[rustfmt::skip]
    let fn_pp: *const fn(#[splat] (u32, i8), f64) = &(splat_non_terminal_arg as _);
    unsafe {
        (*fn_pp)(1, 2, 3.5);
        (*fn_pp)(1u32, 2i8, 3.5f64);
    }
}
