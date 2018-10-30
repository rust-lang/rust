// Optimization kills all the reborrows, enough to make this error go away.  There are
// no retags either because we don't retag immediately after a `&[mut]`; we rely on
// that creating a fresh reference.
// See `shared_confusion_opt.rs` for a variant that is caught even with optimizations.
// Keep this test to make sure that without optimizations, we do not have to actually
// use the `x_inner_shr`.
// compile-flags: -Zmir-opt-level=0

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
        let x_inner_shr = &*x_inner; // freezing again
    }
    // Our old raw should be dead by now
    unsafe { *x_evil = 0; } // this falls back to some Raw higher up the stack
    *x_inner = 12; //~ ERROR reference with non-reactivatable tag
}

fn main() {
    test(&mut RefCell::new(0));
}
