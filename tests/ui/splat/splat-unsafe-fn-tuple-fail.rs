//! Test that using `#[splat]` incorrectly on unsafe functions gives errors.

#![allow(incomplete_features)]
#![feature(splat)]

unsafe fn unsafe_wrong_type(#[splat] _x: u32) {}
//~^ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a u32

unsafe fn unsafe_multi_splat(#[splat] (_a, _b): (u32, i8), #[splat] (_c, _d): (u32, i8)) {}
//~^ ERROR multiple `#[splat]`s are not allowed in the same function

fn main() {
    unsafe {
        unsafe_wrong_type(1u32);
    }
}
