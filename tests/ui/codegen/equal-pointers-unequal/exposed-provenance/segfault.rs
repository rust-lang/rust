//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-fail
//@ check-run-results

// This one should segfault.
// I don't know a better way to check for segfault other than
// check that it fails and that the output is empty.

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1431758601

#![feature(exposed_provenance)]

use std::{cell::RefCell, ptr::addr_of};

fn main() {
    let a = {
        let v = 0u8;
        addr_of!(v).expose_provenance()
    };
    let b = {
        let v = 0u8;
        addr_of!(v).expose_provenance()
    };
    let i = b - a;
    let arr = [
        RefCell::new(Some(Box::new(1))),
        RefCell::new(None),
        RefCell::new(None),
        RefCell::new(None),
    ];
    assert_ne!(i, 0);
    let r = arr[i].borrow();
    let r = r.as_ref().unwrap();
    *arr[0].borrow_mut() = None;
    println!("{}", *r);
}
