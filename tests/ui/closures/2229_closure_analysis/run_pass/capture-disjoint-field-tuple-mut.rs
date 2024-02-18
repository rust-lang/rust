//@ edition:2021
//@ run-pass

// Test that we can mutate an element of a tuple from within a closure
// while immutably borrowing another element of the same tuple outside the closure.

#![feature(rustc_attrs)]

fn main() {
    let mut t = (10, 10);

    let mut c = || {
        let t1 = &mut t.1;
        *t1 = 20;
    };

    // Test that `c` only captures t.1, therefore reading t.0 is allowed.
    println!("{}", t.0);
    c();
}
