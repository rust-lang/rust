//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ check-run-results
//@ normalize-stdout-test: "\d+" -> "<..>"

// Based on https://github.com/rust-lang/rust/issues/107975#issuecomment-1432161340

#![feature(exposed_provenance)]

use std::ptr::addr_of;

#[inline(never)]
fn cmp(a: usize, b: usize) -> bool {
    a == b
}

#[inline(always)]
fn cmp_in(a: usize, b: usize) -> bool {
    a == b
}

fn main() {
    let a = {
        let v = 0;
        addr_of!(v).expose_provenance()
    };
    let b = {
        let v = 0;
        addr_of!(v).expose_provenance()
    };
    assert_eq!(a.to_string(), b.to_string());
    println!("{a:?} == {b:?} -> ==: {}, cmp_in: {}, cmp: {}", a == b, cmp_in(a, b), cmp(a, b));
}
