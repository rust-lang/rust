// compile-flags: -Zunleash-the-miri-inside-of-you

#![allow(const_err)]

use std::cell::UnsafeCell;

// this test ensures that our mutability story is sound

struct Meh {
    x: &'static UnsafeCell<i32>,
}
unsafe impl Sync for Meh {}

// the following will never be ok! no interior mut behind consts, because
// all allocs interned here will be marked immutable.
const MUH: Meh = Meh { //~ ERROR: mutable memory (`UnsafeCell`) is not allowed in constant
    x: &UnsafeCell::new(42),
};

struct Synced {
    x: UnsafeCell<i32>,
}
unsafe impl Sync for Synced {}

// Make sure we also catch this behind a type-erased `dyn Trait` reference.
const SNEAKY: &dyn Sync = &Synced { x: UnsafeCell::new(42) };
//~^ ERROR: mutable memory (`UnsafeCell`) is not allowed in constant

// Make sure we also catch mutable references.
const BLUNT: &mut i32 = &mut 42;
//~^ ERROR: mutable memory (`&mut`) is not allowed in constant

fn main() {
    unsafe {
        *MUH.x.get() = 99;
    }
}
