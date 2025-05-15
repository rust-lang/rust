//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass

// Based on https://github.com/rust-lang/rust/issues/107975#issuecomment-1432161340

use std::ptr;

#[inline(never)]
fn cmp(a: usize, b: usize) -> bool {
    a == b
}

#[inline(always)]
fn cmp_in(a: usize, b: usize) -> bool {
    a == b
}

fn main() {
    let a: usize = {
        let v = 0;
        ptr::from_ref(&v).addr()
    };
    let b: usize = {
        let v = 0;
        ptr::from_ref(&v).addr()
    };
    assert_eq!(format!("{a}"), format!("{b}"));
    assert_eq!(format!("{}", a == b), "true");
    assert_eq!(format!("{}", cmp_in(a, b)), "true");
    assert_eq!(format!("{}", cmp(a, b)), "true");
}
