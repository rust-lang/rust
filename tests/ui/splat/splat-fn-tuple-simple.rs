//@ run-pass
//! Test using `#[splat]` on tuple arguments of simple functions.

#![allow(incomplete_features)]
#![feature(splat)]

fn tuple_args(#[splat] (_a, _b): (u32, i8)) {}

fn splat_non_terminal_arg(#[splat] (_a, _b): (u32, i8), _c: f64) {}

fn main() {
    tuple_args(1, 2);
    // FIXME(splat): should splatted functions be callable with tupled and un-tupled arguments?
    // Add a tupled test for each call if they are.
    //tuple_args((1, 2));

    tuple_args(1, 2);
    tuple_args(1u32, 2i8);

    splat_non_terminal_arg(1, 2, 3.5);
    splat_non_terminal_arg(1u32, 2i8, 3.5f64);

    #[expect(unused_variables, reason = "FIXME(splat or lint): this is obviously used")]
    let fn_ptr = tuple_args;
    fn_ptr(1, 2);
    fn_ptr(1u32, 2i8);

    #[expect(unused_variables, reason = "FIXME(splat or lint): this is obviously used")]
    let fn_ptr = splat_non_terminal_arg;
    fn_ptr(1, 2, 3.5);
    fn_ptr(1u32, 2i8, 3.5f64);
}
