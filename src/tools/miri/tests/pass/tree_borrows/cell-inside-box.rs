//@compile-flags: -Zmiri-tree-borrows
#![feature(box_as_ptr)]
#[path = "../../utils/mod.rs"]
#[macro_use]
mod utils;

use std::cell::UnsafeCell;

pub fn main() {
    let cell = UnsafeCell::new(42);
    let box1 = Box::new(cell);

    unsafe {
        let ptr1: *mut UnsafeCell<i32> = Box::into_raw(box1);
        name!(ptr1);

        let mut box2 = Box::from_raw(ptr1);
        // `ptr2` will be a descendant of `ptr1`.
        let ptr2: *mut UnsafeCell<i32> = Box::as_mut_ptr(&mut box2);
        name!(ptr2);

        // We perform a write through `x`.
        // Because `ptr1` is ReservedIM, a child write will make it transition to Active.
        // Because `ptr2` is ReservedIM, a foreign write doesn't have any effect on it.
        let x = (*ptr1).get();
        *x = 1;

        // We can still read from `ptr2`.
        let val = *(*ptr2).get();
        assert_eq!(val, 1);

        let alloc_id = alloc_id!(ptr1);
        print_state!(alloc_id);
    }
}
