#![allow(dead_code)]

use std::panic::UnwindSafe;
use std::sync::Arc;
use std::cell::RefCell;

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
    assert::<Arc<RefCell<i32>>>();
    //~^ ERROR the type `UnsafeCell<i32>` may contain interior mutability and a
    //~| ERROR the type `UnsafeCell<isize>` may contain interior mutability and a
}
