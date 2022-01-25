// check-pass
// aux-build:option_future.rs
//
// Check that if we promise to not impl what would overlap it doesn't actually overlap

#![feature(rustc_attrs)]

extern crate option_future as lib;
use lib::Future;

trait Termination {}

#[rustc_with_negative_coherence]
impl<E> Termination for Option<E> where E: Sized {}
#[rustc_with_negative_coherence]
impl<F> Termination for F where F: Future + Sized {}

fn main() {}
