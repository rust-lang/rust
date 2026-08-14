//@ known-bug: rust-lang/rust#143205
#![feature(generic_const_exprs)]

struct Bug<A = [(); (1).1]> {
    a: Bug,
}

pub fn main() {}
