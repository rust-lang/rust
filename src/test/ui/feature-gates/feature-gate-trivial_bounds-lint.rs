// run-pass

#![allow(unused)]
#![deny(trivial_bounds)] // Ignored without the trivial_bounds feature flag.

struct A where i32: Copy;

fn main() {}
