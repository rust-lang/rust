// build-pass (FIXME(62277): could be check-pass?)
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
