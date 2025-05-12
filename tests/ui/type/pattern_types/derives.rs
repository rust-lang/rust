//! Check that pattern types don't implement traits of their base automatically.
//! Exceptions are `Clone` and `Copy`, which have builtin impls for pattern types.

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
struct Nanoseconds(NanoI32);
//~^ ERROR: binary operation `==` cannot be applied to type `(i32) is 0..=999999999`

type NanoI32 = crate::pattern_type!(i32 is 0..=999_999_999);

fn main() {
    let x = Nanoseconds(unsafe { std::mem::transmute(42) });
    let y = x.clone();
    if y == x {}
}
