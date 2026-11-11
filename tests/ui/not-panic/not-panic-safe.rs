#![allow(dead_code)]

use std::panic::UnwindSafe;

fn assert<T: UnwindSafe + ?Sized>() {}

fn main() {
    assert::<&mut &mut &i32>();
    //~^ ERROR the type `&mut &mut &i32` may not be safely transferred across an unwind boundary
}
