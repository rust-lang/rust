//@ check-pass
//@ compile-flags: --test
// Test that we can pass a test through a macro_rules! macro that removes the span of the item
// Regression test for https://github.com/rust-lang/rust/issues/161917
#![feature(macro_attr)]

macro_rules! ohno {
    attr() { $(#[$a:meta])* fn $name:ident () $body: block } => {
        $(#[$a])*
        fn $name () $body
    }
}

#[test]
#[ohno]
fn my_test() {}

fn main() {}
