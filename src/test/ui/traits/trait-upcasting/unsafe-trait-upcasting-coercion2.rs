#![feature(trait_upcasting)]
#![allow(incomplete_features)]

trait A {
    fn foo_a(&self) {}
}

trait B {
    fn foo_b(&self) {}
}

trait C: A + B {
    fn foo_c(&self) {}
}

struct S;

impl A for S {}

impl B for S {}

impl C for S {}

fn invalid_coercion(v: * const Box<dyn C>) {
    let g: * const Box<dyn A> = v;
    //~^ ERROR mismatched types
}

fn main() {}
