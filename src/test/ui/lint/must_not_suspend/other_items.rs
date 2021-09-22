// edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend] //~ ERROR attribute should be
mod inner {}

fn main() {}
