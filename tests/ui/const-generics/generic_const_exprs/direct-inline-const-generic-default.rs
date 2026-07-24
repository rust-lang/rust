//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/159063.

#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

struct S<const N: usize = const { 0 }>;

fn main() {}
