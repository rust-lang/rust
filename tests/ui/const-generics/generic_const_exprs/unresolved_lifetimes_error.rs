#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn foo() -> [(); {
    let a: &'a ();
    //~^ ERROR: use of undeclared lifetime name `'a`
    10_usize
}] {
    loop {}
}

fn main() {}
