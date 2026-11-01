//@ check-pass

//! Tests that operators from supertrait are available directly on `self` for an inheritor trait.
//!
//! # Context
//! Original issue: https://github.com/rust-lang/rust/issues/18088

pub trait Indexable<T>: std::ops::Index<usize, Output = T> {
    fn index2(&self, i: usize) -> &T {
        &self[i]
    }
}
fn main() {}
