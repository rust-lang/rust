//! This test verifies that the Signature Version Hash (SVH) system correctly identifies
//! when changes to an auxiliary crate do not affect its public API.
//!
//! Specifically, it checks that adding non-public items to a crate does not alter
//! its SVH, preventing unnecessary recompilations of dependent crates.

//@ run-pass

// Note that these aux-build directives must be in this order

//@ aux-build:svh-a-base.rs
//@ aux-build:svh-b.rs
//@ aux-build:svh-a-base.rs

extern crate a;
extern crate b;

fn main() {
    b::foo()
}
