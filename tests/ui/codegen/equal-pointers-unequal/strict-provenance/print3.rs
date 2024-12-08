//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1430704499

use std::ptr;

fn main() {
    let a: usize = {
        let v = 0;
        ptr::from_ref(&v).addr()
    };
    let b: usize = {
        let v = 0;
        ptr::from_ref(&v).addr()
    };

    assert_ne!(a, b);
    assert_ne!(a, b);
    let c = a;
    assert_eq!(format!("{} {} {}", a == b, a == c, b == c), "false true false");
    println!("{a} {b}");
    assert_eq!(format!("{} {} {}", a == b, a == c, b == c), "true true true");
}
