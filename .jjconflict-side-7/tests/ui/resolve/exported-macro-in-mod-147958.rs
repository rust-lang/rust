//! Regression test for <https://github.com/rust-lang/rust/issues/147958>

//@ check-pass

#![feature(decl_macro)]

macro_rules! exported {
    () => {
        #[macro_export]
        macro_rules! exported {
            () => {};
        }
    };
}
use inner1::*;
exported!();
mod inner1 {
    pub macro exported() {}
}

fn main() {}
