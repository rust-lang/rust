//! This test is very similar to `mut_mismatch3`, but actually writes to the static.
//! In case we relaxed `mut_mismatch3` UB, we still want this to remain UB.

#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;

#[no_mangle]
static IMMUT: i32 = 42;

#[no_mangle]
static INTERIOR_MUT: SyncUnsafeCell<i32> = SyncUnsafeCell::new(42);

fn main() {
    unsafe {
        extern "C" {
            static mut INTERIOR_MUT: i32;
        }
        (&raw mut INTERIOR_MUT).write(7);
    }

    unsafe {
        extern "C" {
            static mut IMMUT: i32;
        }
        (&raw mut IMMUT).write(7);
        //~^ ERROR: is declared as an mutable `static`, but the backing static is immutable
    }
}
