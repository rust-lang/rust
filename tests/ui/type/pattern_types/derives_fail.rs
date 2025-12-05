//! Check that pattern types don't implement traits of their base automatically.
//! Exceptions are `Clone` and `Copy`, which have bultin impls for pattern types.

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
struct Nanoseconds(NanoI32);
//~^ ERROR: the trait bound `(i32) is 0..=999999999: Eq` is not satisfied
//~| ERROR: `(i32) is 0..=999999999` doesn't implement `Debug`
//~| ERROR: the trait bound `(i32) is 0..=999999999: Ord` is not satisfied
//~| ERROR: the trait bound `(i32) is 0..=999999999: Hash` is not satisfied
//~| ERROR: the trait bound `(i32) is 0..=999999999: Default` is not satisfied
//~| ERROR: can't compare `(i32) is 0..=999999999` with `_`
//~| ERROR: `==` cannot be applied

type NanoI32 = crate::pattern_type!(i32 is 0..=999_999_999);

fn main() {
    let x = Nanoseconds(unsafe { std::mem::transmute(42) });
    let y = x.clone();
    if y == x {}
}
