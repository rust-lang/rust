//! Verify that a local `extern type` can manually implement `Unpin` with `pin_ergonomics` enabled.
//! Unlike a `#[pin_v2]` ADT, an extern type has no fields that could be structually pinned.
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
