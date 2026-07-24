//! Test that using `#[arg_splat]` incorrectly on unsafe functions gives errors.

#![allow(incomplete_features)]
#![feature(arg_splat)]

unsafe fn unsafe_wrong_type(#[arg_splat] _x: u32) {}
//~^ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a u32

unsafe fn unsafe_multi_splat(#[arg_splat] (_a, _b): (u32, i8), #[arg_splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[arg_splat]`s are not allowed in the same function argument list

fn main() {
    unsafe {
        unsafe_wrong_type(1u32);
    }
}
