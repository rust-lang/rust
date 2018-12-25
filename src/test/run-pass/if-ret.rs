#![allow(unused_parens)]
// pretty-expanded FIXME #23616

fn foo() { if (return) { } }

pub fn main() { foo(); }
