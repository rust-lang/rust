//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1430704499

use std::ptr;

fn main() {
    let a: usize = {
        let v = 0;
        ptr::from_ref(&v).expose_provenance()
    };
    let b: usize = {
        let v = 0;
        ptr::from_ref(&v).expose_provenance()
    };

    assert_ne!(a, b);
    println!("{a}"); // or b
    assert_eq!(a, b);
}
