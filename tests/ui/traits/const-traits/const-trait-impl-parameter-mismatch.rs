// This test demonstrates an ICE that may occur when we try to resolve the instance
// of a impl that has different generics than the trait it's implementing. This ensures
// we first check that the args are compatible before resolving the body, just like
// we do in projection before substituting a GAT.
//
// Regression test for issue #125877.

//@ compile-flags: -Znext-solver
//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![feature(const_trait_impl, effects)]
//~^ ERROR feature has been removed

#[const_trait]
trait Main {
    fn compute<T: ~const Aux>() -> u32;
}

impl const Main for () {
    fn compute<'x>() -> u32 {
        //~^ ERROR associated function `compute` has 0 type parameters but its trait declaration has 1 type parameter
        0
    }
}

#[const_trait]
trait Aux {}

impl const Aux for () {}

fn main() {
    const _: u32 = <()>::compute::<()>();
}
