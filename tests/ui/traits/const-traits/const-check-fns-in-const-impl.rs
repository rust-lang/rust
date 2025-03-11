//@ compile-flags: -Znext-solver

#![feature(const_trait_impl)]

struct S;
#[const_trait]
trait T {
    (const) fn foo();
}

fn non_const() {}

impl const T for S {
    (const) fn foo() { non_const() }
    //~^ ERROR cannot call non-const function
}

fn main() {}
