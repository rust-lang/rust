// This test demonstrates an ICE that may occur when we try to resolve the instance
// of a impl that has different generics than the trait it's implementing. This ensures
// we first check that the args are compatible before resolving the body, just like
// we do in projection before substituting a GAT.
//
// Const traits aren't the only way to achieve this ICE, but it's a convenient way
// to ensure the inliner is called.

//@ compile-flags: -Znext-solver -Zinline-mir=yes

#![feature(const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete

trait Trait {
    fn foo(self);
}

impl Trait for () {
    #[inline]
    fn foo<T>(self) {
        //~^ ERROR  method `foo` has 1 type parameter but its trait declaration has 0 type parameters
        todo!();
    }
}

const fn foo() {
    ().foo();
}

const UWU: () = foo();

fn main() {}
