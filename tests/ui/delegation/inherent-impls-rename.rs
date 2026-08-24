//@ check-pass

#![feature(fn_delegation)]

struct X;

fn foo() {}

impl X {
    reuse foo as bar;
}

reuse X::bar;

fn main() {}
