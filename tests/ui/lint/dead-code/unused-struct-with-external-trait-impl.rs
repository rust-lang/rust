//! An unused struct implementing an external trait such as `FromStr`
//! should not be considered live.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/142541>.
#![deny(dead_code)]

impl std::str::FromStr for Foo {
    type Err = ();
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        panic!();
    }
}

struct Foo; //~ ERROR: struct `Foo` is never constructed

fn main() {}
