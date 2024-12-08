//@ known-bug: #114317
#![feature(generic_const_exprs)]

struct A<const B: str = 1, C>;

fn main() {}
