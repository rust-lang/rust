//@compile-flags: -Znext-solver=globally
#![feature(inherent_associated_types)]

struct Foo;
impl<const X: y> Foo {
    //~^ ERROR the const parameter `X` is not constrained by the impl trait, self type, or predicates
    //~| ERROR cannot find type `y` in this scope
    type ImplTrait = impl Clone;
    //~^ ERROR `impl Trait` in associated types is unstable
    //~| ERROR unconstrained opaque type
    fn f() -> Self::ImplTrait {
        ()
        //~^ ERROR mismatched types
    }
}
//~^ ERROR `main` function not found
