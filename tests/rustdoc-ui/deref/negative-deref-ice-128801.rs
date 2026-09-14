//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/128801
// Negative `Deref`/`DerefMut` impls should not cause an ICE.

#![feature(negative_impls)]

pub struct Source;

impl !std::ops::Deref for Source {}
impl !std::ops::DerefMut for Source {}
