//@ edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend] //~ ERROR attribute cannot be used on modules
mod inner {}

fn main() {}
