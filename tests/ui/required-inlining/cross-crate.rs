//@ aux-build:callees.rs
//@ build-pass
//@ compile-flags: --crate-type=lib
#![feature(required_inlining)]

extern crate callees;

// Test that required inlining across crates works as expected.

pub fn caller() {
    callees::required();

    callees::must();
}
