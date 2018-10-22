#![allow(unused_variables)]
use std::cell::RefCell;

fn test(r: &mut RefCell<i32>) {
    let x = &*r; // not freezing because interior mutability
    let mut x_ref = x.borrow_mut();
    let x_inner : &mut i32 = &mut *x_ref; // Uniq reference
    let x_evil = x_inner as *mut _;
    {
        let x_inner_shr = &*x_inner; // frozen
        let y = &*r; // outer ref, not freezing
        let x_inner_shr2 = &*x_inner; // freezing again
    }
    // Our old raw should be dead by now
    unsafe { *x_evil = 0; } // this falls back to some Raw higher up the stack
    *x_inner = 12; //~ ERROR does not exist on the stack
}

fn main() {
    test(&mut RefCell::new(0));
}
