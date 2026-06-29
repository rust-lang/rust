//@ run-pass
//@ edition:2024
//! Test using `#[splat]` on tuple arguments of async functions.

#![allow(incomplete_features)]
#![feature(splat)]

async fn async_tuple_args(#[splat] (_a, _b): (u32, i8)) {}

async fn async_splat_non_terminal_arg(#[splat] (_a, _b): (u32, i8), _c: f64) {}

fn main() {
    let _ = async_tuple_args(1u32, 2i8);
    let _ = async_tuple_args(1, 2);

    let _ = async_splat_non_terminal_arg(1u32, 2i8, 3.5f64);
    let _ = async_splat_non_terminal_arg(1, 2, 3.5);
}
