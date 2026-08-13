// Regression test for <https://github.com/rust-lang/rust/issues/129765>.
// A `#[test_case]` attribute on a `use` item used to ICE with
// "TyKind::Error constructed but no error reported".
//@ check-pass
//@ compile-flags: --test

#![feature(custom_test_frameworks)]
#![test_runner(crate::test_runner)]
#![allow(unused_imports)]

pub fn test_runner(tests: &[&dyn Fn()]) {
    for test in tests {
        test();
    }
}

#[test_case]
use std::mem;

fn main() {}
