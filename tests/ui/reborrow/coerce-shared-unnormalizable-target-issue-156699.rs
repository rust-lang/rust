//! Regression test for <https://github.com/rust-lang/rust/issues/156699>.
//!
//! The target type of a `CoerceShared` impl that could not be normalized
//! used to cause an ICE instead of an error.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct Wrapper<'a, T>(&'a mut T);

impl<'a, T> Reborrow for Wrapper<'a, T> {}

impl<'a, T> CoerceShared<<T as Iterator>::Item> for Wrapper<'a, T> {}
//~^ ERROR `T` is not an iterator

fn main() {}
