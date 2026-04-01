//@ check-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]

struct Foo<const A: [(); 0 + 0]> where [(); 0 + 0]: Sized;

fn main() {}
