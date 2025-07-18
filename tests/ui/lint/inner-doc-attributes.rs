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
