// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

use std::fmt::Debug;

#[derive(Debug)]
struct S<T: Debug, const N: usize>([T; N]);

fn main() {}
