//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// This test is a repro for #82279. It checks that we don't error
// when calling is_const_evaluatable on `std::mem::size_of::<T>()`
// when looking for candidates that may prove `T: Foo` in `foo`

trait Foo {}

#[allow(dead_code)]
fn foo<T: Foo>() {}

impl<T> Foo for T where [(); std::mem::size_of::<T>()]:  {}

fn main() {}
