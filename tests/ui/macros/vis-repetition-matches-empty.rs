//! Regression test for <https://github.com/rust-lang/rust/issues/42755>.
//! Test `$($v:vis)*` is disallowed.

macro_rules! foo {
    ($($p:vis)*) => {} //~ ERROR repetition matches empty token tree
}

foo!(a);

fn main() {}
