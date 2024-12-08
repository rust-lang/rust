//@ run-pass

#![allow(unused_parens)]

fn foo() { if (return) { } } //~ WARNING unreachable block in `if`

pub fn main() { foo(); }
