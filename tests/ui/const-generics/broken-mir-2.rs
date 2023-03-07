// run-pass

#![allow(dead_code)]

use std::fmt::Debug;

#[derive(Debug)]
struct S<T: Debug, const N: usize>([T; N]);

fn main() {}
