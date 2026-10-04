//! Regression test for https://github.com/rust-lang/rust/issues/10853

//@ check-pass

#![deny(missing_docs)]
#![doc="module"]

#[doc="struct"]
pub struct Foo;

pub fn foo() {
    #![doc="fn"]
}

#[doc="main"]
pub fn main() {}
