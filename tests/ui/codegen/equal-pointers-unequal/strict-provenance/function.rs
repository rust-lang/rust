//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

// Based on https://github.com/rust-lang/rust/issues/107975#issuecomment-1434203908

use std::ptr;

fn f() -> usize {
    let v = 0;
    ptr::from_ref(&v).addr()
}

fn main() {
    let a = f();
    let b = f();

    // `a` and `b` are not equal.
    assert_ne!(a, b);
    // But they are the same number.
    assert_eq!(format!("{a}"), format!("{b}"));
    // And they are equal.
    assert_eq!(a, b);
}
