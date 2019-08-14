// run-pass
#![allow(unused_variables)]

#![feature(generators)]

use std::cell::RefCell;

struct A;

impl A {
    fn test(&self, a: ()) {}
}

fn main() {
    // Test that the MIR local with type &A created for the auto-borrow adjustment
    // is caught by typeck
    move || {
        A.test(yield);
    };

    // Test that the std::cell::Ref temporary returned from the `borrow` call
    // is caught by typeck
    let y = RefCell::new(true);
    static move || {
        yield *y.borrow();
        return "Done";
    };
}
