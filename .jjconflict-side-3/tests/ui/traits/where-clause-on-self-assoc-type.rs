//! Regression test for <https://github.com/rust-lang/rust/issues/21140>.
//! Tests we don't ICE on where clauses on self associated type.

//@ check-pass
pub trait Trait where Self::Out: std::fmt::Display {
    type Out;
}

fn main() {}
