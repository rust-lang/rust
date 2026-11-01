// Regression test for https://github.com/rust-lang/rust/issues/151307.
//@ compile-flags: -Znext-solver
#![feature(return_type_notation)]

trait Trait {
    fn foo() -> impl Copy {}
}

impl Trait for () {
    fn foo() -> impl Copy
    //~^ ERROR cycle detected
    where
        Self::foo(..): Copy,
    {
    }
}

fn main() {}
