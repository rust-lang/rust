//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

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

    // `a` and `b` are not equal.
    assert_ne!(a, b);
    // But they are the same number.
    assert_eq!(format!("{a}"), format!("{b}"));
    // And they are equal.
    assert_eq!(a, b);
}
