//@ edition: 2021

use core::hint::black_box;

#[derive(Debug, PartialEq)]
struct Foo(u32);

fn main() {
    assert_ne!(
        black_box(Foo(5)), // Make sure this expression's span isn't lost.
        if black_box(false) {
            Foo(0) //
        } else {
            Foo(1) //
        }
    );
    ()
}

// This test is a short fragment extracted from `issue-84561.rs`, highlighting
// a particular span of code that can easily be lost if overlapping spans are
// processed incorrectly.
