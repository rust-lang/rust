// issue: <https://github.com/rust-lang/rust/issues/11592>
// Test that the `missing_docs` lint does not trigger for a private trait.
//@ check-pass
//! Ensure the private trait Bar isn't complained about.

#![deny(missing_docs)]

mod foo {
    trait Bar {
        fn bar(&self) {}
    }
    impl Bar for i8 {
        fn bar(&self) {}
    }
}

fn main() {}
