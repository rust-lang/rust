#![feature(unsize)]

use std::marker::Unsize;

pub struct MyType;

impl<T> Unsize<T> for MyType {}
//~^ ERROR explicit impls for the `Unsize` trait are not permitted [E0328]

fn main() {}
