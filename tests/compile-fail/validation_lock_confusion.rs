// Make sure validation can handle many overlapping shared borrows for different parts of a data structure
#![allow(unused_variables)]
use std::cell::RefCell;

fn evil(x: *mut i32) {
    unsafe { *x = 0; } //~ ERROR: in conflict with lock WriteLock
}

fn test(r: &mut RefCell<i32>) {
    let x = &*r; // releasing write lock, first suspension recorded
    let mut x_ref = x.borrow_mut();
    let x_inner : &mut i32 = &mut *x_ref; // new inner write lock, with same lifetime as outer lock
    {
        let x_inner_shr = &*x_inner; // releasing inner write lock, recording suspension
        let y = &*r; // second suspension for the outer write lock
        let x_inner_shr2 = &*x_inner; // 2nd suspension for inner write lock
    }
    // If the two locks are mixed up, here we should have a write lock, but we do not.
    evil(x_inner as *mut _);
}

fn main() {
    test(&mut RefCell::new(0));
}
