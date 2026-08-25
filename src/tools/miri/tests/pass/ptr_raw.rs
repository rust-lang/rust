use std::mem;
use std::ptr::{self, addr_of};

fn basic_raw() {
    let mut x = 12;
    let x = &mut x;

    assert_eq!(*x, 12);

    let raw = x as *mut i32;
    unsafe {
        *raw = 42;
    }

    assert_eq!(*x, 42);

    let raw = x as *mut i32;
    unsafe {
        *raw = 12;
    }
    *x = 23;

    assert_eq!(*x, 23);
}

fn assign_overlapping() {
    // Test an assignment where LHS and RHS alias.
    // In Mir, that's UB (see `fail/overlapping_assignment.rs`), but in surface Rust this is allowed.
    let mut mem = [0u32; 4];
    let ptr = &mut mem as *mut [u32; 4];
    unsafe { *ptr = *ptr };
}

fn deref_invalid() {
    unsafe {
        // `addr_of!(*ptr)` is never UB.
        let _val = addr_of!(*ptr::without_provenance::<i32>(0));
        let _val = addr_of!(*ptr::without_provenance::<i32>(1)); // not aligned

        // Similarly, just mentioning the place is fine.
        let _ = *ptr::without_provenance::<i32>(0);
        let _ = *ptr::without_provenance::<i32>(1);
    }
}

fn deref_partially_dangling() {
    let x = (1, 13);
    let xptr = &x as *const _ as *const (i32, i32, i32);
    let val = unsafe { (*xptr).1 };
    assert_eq!(val, 13);
}

fn deref_too_big_slice() {
    unsafe {
        let slice: *const [u8] = mem::transmute((1usize, usize::MAX));
        // `&*slice` would complain that the slice is too big, but in a raw pointer this is fine.
        let _val = addr_of!(*slice);
    }
}

fn main() {
    basic_raw();
    assign_overlapping();
    deref_invalid();
    deref_partially_dangling();
    deref_too_big_slice();
}
