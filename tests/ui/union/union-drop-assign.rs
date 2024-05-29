//@ run-pass
#![allow(unused_assignments)]

// Drop works for union itself.

use std::mem::ManuallyDrop;

struct S;

union U {
    a: ManuallyDrop<S>
}

impl Drop for S {
    fn drop(&mut self) {
        unsafe { CHECK += 10; }
    }
}

impl Drop for U {
    fn drop(&mut self) {
        unsafe { CHECK += 1; }
    }
}

static mut CHECK: u8 = 0;

fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK, 0);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK, 1); // union itself is assigned, union is dropped, field is not dropped
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
        *u.a = S;
        assert_eq!(CHECK, 11); // union field is assigned, field is dropped
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}
