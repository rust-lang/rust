#![feature(plugin, never_type)]
#![plugin(clippy)]
#![deny(diverging_sub_expression)]

#[allow(empty_loop)]
fn diverge() -> ! { loop {} }

struct A;

impl A {
    fn foo(&self) -> ! { diverge() }
}

#[allow(unused_variables, unnecessary_operation)]
fn main() {
    let b = true;
    b || diverge(); //~ ERROR sub-expression diverges
    b || A.foo(); //~ ERROR sub-expression diverges
    let y = (5, diverge(), 6); //~ ERROR sub-expression diverges
    println!("{}", y.1);
}
