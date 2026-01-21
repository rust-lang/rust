// Regression test for <https://github.com/rust-lang/rust/issues/49593>.
//
// This checks that we can construct `Box<dyn Error>` by calling `Box::new`
// with a value of the never type. And similarly for raw pointers.
//
// This used to fail because we tried to coerce `! -> dyn Error`, which then
// failed because we were trying to pass an unsized value by value, etc.
//
// On edition <= 2021 this currently fails because of never type fallback to
// unit.
//
//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//
//@[e2024] check-pass

#![feature(never_type)]

use std::error::Error;
use std::mem;

fn raw_ptr<T>(t: T) -> *mut T {
    panic!()
}

fn foo(x: !) -> Box<dyn Error> {
    Box::new(x)
    //[e2021]~^ ERROR trait bound `(): std::error::Error` is not satisfied
}

fn foo_raw_ptr(x: !) -> *mut dyn Error {
    raw_ptr(x)
    //[e2021]~^ ERROR trait bound `(): std::error::Error` is not satisfied
}

fn main() {}
