//! We want to reserve rights to be able to optimize statics declared as immutable,
//! so we defensively disallow immutable statics pointing to mutable allocations.

#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;

#[export_name = "S"]
static INTERIOR_MUT_S: SyncUnsafeCell<i32> = SyncUnsafeCell::new(42);

fn main() {
    extern "C" {
        static S: i32;
    }
    let _val = &raw const S;
    //~^ ERROR: is declared as an immutable `static`, but the backing static is mutable
}
