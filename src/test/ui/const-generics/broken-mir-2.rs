#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

use std::fmt::Debug;

#[derive(Debug)]
struct S<T: Debug, const N: usize>([T; N]);
//~^ ERROR arrays only have std trait implementations for lengths 0..=32

fn main() {}
