#![allow(dead_code)]

use std::panic::UnwindSafe;
use std::cell::UnsafeCell;

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
    assert::<*const UnsafeCell<i32>>(); //~ ERROR E0277
}
