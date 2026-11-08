//@ compile-flags: -Znext-solver
// Regression test for https://github.com/rust-lang/rust/issues/151307.
// This used to ICE.
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
