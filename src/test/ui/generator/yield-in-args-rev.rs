// run-pass
#![allow(dead_code)]

// Test that a borrow that occurs after a yield in the same
// argument list is not treated as live across the yield by
// type-checking.

#![feature(generators)]

fn foo(_a: (), _b: &bool) {}

fn bar() {
    || {
        let b = true;
        foo(yield, &b);
    };
}

fn main() { }
