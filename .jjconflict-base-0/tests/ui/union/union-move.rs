//! Test the behavior of moving out of non-`Copy` union fields.
//! Avoid types that `Drop`, we want to focus on moving.

use std::cell::RefCell;
use std::mem::ManuallyDrop;

fn move_out<T>(x: T) {}

union U1 {
    f1_nocopy: ManuallyDrop<RefCell<i32>>,
    f2_nocopy: ManuallyDrop<RefCell<i32>>,
    f3_copy: i32,
}

union U2 {
    f1_nocopy: ManuallyDrop<RefCell<i32>>,
}
impl Drop for U2 {
    fn drop(&mut self) {}
}

fn test1(x: U1) {
    // Moving out of a nocopy field prevents accessing other nocopy field.
    unsafe {
        move_out(x.f1_nocopy);
        move_out(x.f2_nocopy); //~ ERROR use of moved value: `x`
    }
}

fn test2(x: U1) {
    // "Moving" out of copy field doesn't prevent later field accesses.
    unsafe {
        move_out(x.f3_copy);
        move_out(x.f2_nocopy); // no error
    }
}

fn test3(x: U1) {
    // Moving out of a nocopy field prevents accessing other copy field.
    unsafe {
        move_out(x.f2_nocopy);
        move_out(x.f3_copy); //~ ERROR use of moved value: `x`
    }
}

fn test4(x: U2) {
    // Cannot move out of union that implements `Drop`.
    unsafe {
        move_out(x.f1_nocopy); //~ ERROR cannot move out of type `U2`, which implements the `Drop`
    }
}

fn main() {}
