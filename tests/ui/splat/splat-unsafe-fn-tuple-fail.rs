//! Test that using `#[rustc_splat]` incorrectly on unsafe functions gives errors.

#![allow(incomplete_features)]
#![feature(splat)]

unsafe fn unsafe_wrong_type(#[rustc_splat] _x: u32) {}
//~^ ERROR cannot use `rustc_splat` attribute; the splatted argument type must be a tuple or unit, not a u32

unsafe fn unsafe_multi_splat(
    #[rustc_splat] (_a, _b): (u32, i8),
    //~^ ERROR multiple `#[rustc_splat]`s are not allowed in the same function argument list
    #[rustc_splat] (_c, _d): (u32, i8),
) {}

fn main() {
    unsafe {
        unsafe_wrong_type(1u32);
    }
}
