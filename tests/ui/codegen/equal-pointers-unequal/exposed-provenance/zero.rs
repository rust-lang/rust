//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ ignore-backends: gcc

// Derived from https://github.com/rust-lang/rust/issues/107975#issuecomment-1431758601

use std::ptr;

fn main() {
    let a: usize = {
        let v = 0u8;
        ptr::from_ref(&v).expose_provenance()
    };
    let b: usize = {
        let v = 0u8;
        ptr::from_ref(&v).expose_provenance()
    };

    // So, are `a` and `b` equal?

    // Let's check their difference.
    let i: usize = a - b;
    // It's not zero, which means `a` and `b` are not equal.
    assert_ne!(i, 0);
    // But it looks like zero...
    assert_eq!(format!("{i}"), "0");
    // ...and now it *is* zero?
    assert_eq!(i, 0);
    // So `a` and `b` are equal after all?
}
