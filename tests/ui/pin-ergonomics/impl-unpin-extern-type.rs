//! Check that a manual `Unpin` impl for a local `extern type` is accepted under
//! `pin_ergonomics`. An `extern type` has no fields, so it can never be structurally
//! pinned, unlike a `#[pin_v2]` ADT.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/155053>.

//@ check-pass

#![feature(pin_ergonomics)]
#![feature(extern_types)]
#![allow(incomplete_features)]

unsafe extern "C" {
    type ExternType;
}

impl Unpin for ExternType {}

fn main() {}
