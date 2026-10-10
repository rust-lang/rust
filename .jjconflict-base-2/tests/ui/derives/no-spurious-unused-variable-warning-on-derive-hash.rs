//! Regression test for https://github.com/rust-lang/rust/issues/32292
//@ run-pass
#![deny(warnings)]

#[derive(Hash, Ord, PartialOrd, Eq, PartialEq, Debug, Clone, Copy)]
struct Foo;

fn main() {
    let _ = Foo;
}
