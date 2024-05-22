//@ run-pass
#![allow(dead_code)]

// Test that a borrow that occurs after a yield in the same
// argument list is not treated as live across the yield by
// type-checking.

#![feature(coroutines)]

fn foo(_a: (), _b: &bool) {}

fn bar() {
    #[coroutine] || { //~ WARN unused coroutine that must be used
        let b = true;
        foo(yield, &b);
    };
}

fn main() { }
