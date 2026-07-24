//@ edition:2024
//! Test that using `#[arg_splat]` incorrectly on async functions gives errors.

#![allow(incomplete_features)]
#![feature(arg_splat)]

async fn async_wrong_type(#[arg_splat] _x: u32) {}
//~^ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a u32

async fn async_multi_splat(#[arg_splat] (_a, _b): (u32, i8), #[arg_splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[arg_splat]`s are not allowed in the same function argument list

fn main() {
    async_wrong_type(1u32);
    async_multi_splat(1u32, 2i8, 3u32, 4i8);
    //~^ ERROR this splatted function takes 3 arguments, but 4 were provided
}
