#![feature(fn_delegation)]

struct X;

fn foo() {}

impl X {
    reuse foo as bar;
}

reuse X::bar;
//~^ ERROR: cannot find function `bar` in `X`

fn main() {}
