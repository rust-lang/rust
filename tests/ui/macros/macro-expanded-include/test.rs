//@ needs-asm-support
//@ check-pass
#![allow(unused)]

#[macro_use]
mod foo;

m!();
fn f() {
    n!();
}

fn main() {}
