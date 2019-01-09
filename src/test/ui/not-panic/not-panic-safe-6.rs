#![allow(dead_code)]

use std::panic::UnwindSafe;
use std::cell::RefCell;

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
    assert::<*mut RefCell<i32>>();
    //~^ ERROR the type `std::cell::UnsafeCell<i32>` may contain interior mutability and a
    //~| ERROR the type `std::cell::UnsafeCell<isize>` may contain interior mutability and a
}
