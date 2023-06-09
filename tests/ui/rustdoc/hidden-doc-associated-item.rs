// check-pass
// See issue #85526.
// This test should produce no warnings.

#![deny(missing_docs)]
//! Crate docs

#[doc(hidden)]
pub struct Foo;

impl Foo {
    pub fn bar() {}
}

fn main() {}
