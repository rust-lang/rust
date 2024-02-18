//@ check-pass

#![feature(decl_macro)]
macro foo {
    () => {},
}

fn main() {}
