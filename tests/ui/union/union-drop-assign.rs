//@ run-pass
#![allow(unused_assignments)]

// Drop works for union itself.

use std::mem::ManuallyDrop;

struct S;

union U {
    a: ManuallyDrop<S>
}

static mut CHECK: u8 = 0;

fn add_to_check(value: u8) {
    unsafe {
        let check = &raw mut CHECK;
        check.write(check.read() + value);
    }
}

fn check() -> u8 {
    unsafe { (&raw const CHECK).read() }
}

impl Drop for S {
    fn drop(&mut self) {
        add_to_check(10);
    }
}

impl Drop for U {
    fn drop(&mut self) {
        add_to_check(1);
    }
}

fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        assert_eq!(check(), 0);
        u = U { a: ManuallyDrop::new(S) };
        // The union itself is assigned and dropped, but its field is not.
        assert_eq!(check(), 1);
        *u.a = S;
        // Assigning the union field drops the old field value.
        assert_eq!(check(), 11);
    }
}
