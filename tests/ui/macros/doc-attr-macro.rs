//! Regression test for <https://github.com/rust-lang/rust/issues/44730>.
//! Test `#[doc = $expr]` generates docs.
//@ check-pass
//! dox

#![deny(missing_docs)]

macro_rules! doc {
    ($e:expr) => (
        #[doc = $e]
        pub struct Foo;
    )
}

doc!("a");

fn main() {}
