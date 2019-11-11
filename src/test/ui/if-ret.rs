// run-pass

#![allow(unused_parens)]
// pretty-expanded FIXME #23616

fn foo() { if (return) { } } //~ WARNING unreachable block in `if` expression

pub fn main() { foo(); }
