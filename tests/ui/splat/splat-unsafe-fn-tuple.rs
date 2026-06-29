//@ run-pass
//! Test using `#[splat]` on tuple arguments of unsafe functions.

#![allow(incomplete_features)]
#![feature(splat)]

unsafe fn unsafe_tuple_args(#[splat] (_a, _b): (u32, i8)) {}

unsafe fn unsafe_splat_non_terminal_arg(#[splat] (_a, _b): (u32, i8), _c: f64) {}

fn main() {
    unsafe {
        unsafe_tuple_args(1u32, 2i8);
        unsafe_tuple_args(1, 2);

        unsafe_splat_non_terminal_arg(1u32, 2i8, 3.5f64);
        unsafe_splat_non_terminal_arg(1, 2, 3.5);
    }
}
