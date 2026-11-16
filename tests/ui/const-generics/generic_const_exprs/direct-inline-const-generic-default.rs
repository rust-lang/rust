//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/159063.

#![feature(generic_const_exprs)]
#![feature(gca_min_const_items)]

struct S<const N: usize = const { 0 }>;

fn main() {}
