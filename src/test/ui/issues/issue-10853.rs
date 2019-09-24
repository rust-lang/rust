// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

#![deny(missing_docs)]
#![doc="module"]

#[doc="struct"]
pub struct Foo;

pub fn foo() {
    #![doc="fn"]
}

#[doc="main"]
pub fn main() {}
