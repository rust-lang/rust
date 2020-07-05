// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

use std::fmt::Debug;

#[derive(Debug)]
struct S<T: Debug, const N: usize>([T; N]);

fn main() {}
