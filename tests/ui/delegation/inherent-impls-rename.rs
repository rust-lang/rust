//@ check-pass

#![feature(fn_delegation)]

struct X<'a, T>(&'a T);

fn foo() {}

impl X<'_, String> {
    reuse foo as bar;
}

reuse X::<'static, String>::bar;

fn main() {}
