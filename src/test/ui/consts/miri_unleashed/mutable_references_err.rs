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
const MUH: Meh = Meh { //~ ERROR: it is undefined behavior to use this value
    x: &UnsafeCell::new(42),
};

struct Synced {
    x: UnsafeCell<i32>,
}
unsafe impl Sync for Synced {}

// Make sure we also catch this behind a type-erased `dyn Trait` reference.
const SNEAKY: &dyn Sync = &Synced { x: UnsafeCell::new(42) };
//~^ ERROR: it is undefined behavior to use this value

// Make sure we also catch mutable references.
const BLUNT: &mut i32 = &mut 42;
//~^ ERROR: it is undefined behavior to use this value

fn main() {
    unsafe {
        *MUH.x.get() = 99;
    }
}
