//@ stderr-per-bitwidth
//@ compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(invalid_reference_casting, static_mut_refs)]

use std::cell::UnsafeCell;

// this test ensures that our mutability story is sound

struct Meh {
    x: &'static UnsafeCell<i32>,
}
unsafe impl Sync for Meh {}

// the following will never be ok! no interior mut behind consts, because
// all allocs interned here will be marked immutable.
const MUH: Meh = Meh {
    //~^ ERROR: mutable pointer in final value
    x: &UnsafeCell::new(42),
};

fn main() {
    unsafe {
        *MUH.x.get() = 99;
    }
}
