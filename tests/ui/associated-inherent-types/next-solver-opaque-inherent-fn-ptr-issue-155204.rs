//! Regression test for <https://github.com/rust-lang/rust/issues/155204>
//@compile-flags: -Znext-solver=globally
#![feature(inherent_associated_types)]
pub struct Windows<T> {}
//~^ ERROR type parameter `T` is never used

impl<T> Windows<fn(&())> {
    //~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates
    type AssocType = impl Send;
    //~^ ERROR `impl Trait` in associated types is unstable
    //~| ERROR unconstrained opaque type
    fn ret(&self) -> Self::AssocType {
        //~^ ERROR type annotations needed
        ()
        //~^ ERROR mismatched types
    }
}
//~^ ERROR `main` function not found
