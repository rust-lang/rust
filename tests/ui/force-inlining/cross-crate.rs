//@ aux-build:callees.rs
//@ build-pass
//@ compile-flags: --crate-type=lib

extern crate callees;

// Test that forced inlining across crates works as expected.

pub fn caller() {
    callees::forced();

    callees::forced_with_reason();
}
