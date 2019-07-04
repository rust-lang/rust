// build-pass (FIXME(62277): could be check-pass?)
//! Ensure the private trait Bar isn't complained about.

#![deny(missing_docs)]

mod foo {
    trait Bar { fn bar(&self) { } }
    impl Bar for i8 { fn bar(&self) { } }
}

fn main() { }
