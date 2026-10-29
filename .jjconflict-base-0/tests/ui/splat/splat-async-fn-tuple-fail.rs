//@ edition:2024
//! Test that using `#[rustc_splat]` incorrectly on async functions gives errors.

#![allow(incomplete_features)]
#![feature(splat)]

async fn async_wrong_type(#[rustc_splat] _x: u32) {}
//~^ ERROR cannot use `rustc_splat` attribute; the splatted argument type must be a tuple or unit, not a u32

async fn async_multi_splat(
    #[rustc_splat] (_a, _b): (u32, i8),
    //~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list
    #[rustc_splat] (_c, _d): (u32, i8),
) {}

fn main() {
    async_wrong_type(1u32);
    async_multi_splat(1u32, 2i8, 3u32, 4i8);
    //~^ ERROR this splatted function takes 3 arguments, but 4 were provided
}
