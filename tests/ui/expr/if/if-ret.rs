//@ check-pass

#![allow(unused_parens)]

fn foo() { if (return) { } }

pub fn main() { foo(); }
