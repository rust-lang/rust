//@ revisions: re_a re_b re_c

#![cfg_attr(any(), re_a, re_b, re_c)]

//@ aux-build:coherence_lib.rs

// Test that the `Pair` type reports an error if it contains type
// parameters, even when they are covered by local types. This test
// was originally intended to test the opposite, but the rules changed
// with RFC 1023 and this became illegal.

extern crate coherence_lib as lib;
use lib::{Remote,Pair};

pub struct Cover<T>(T);

#[cfg(any(re_a))]
impl<T> Remote for Pair<T,Cover<T>> { }
//[re_a]~^ ERROR E0117

#[cfg(any(re_b))]
impl<T> Remote for Pair<Cover<T>,T> { }
//[re_b]~^ ERROR E0117

#[cfg(any(re_c))]
impl<T,U> Remote for Pair<Cover<T>,U> { }
//[re_c]~^ ERROR E0117

fn main() { }
