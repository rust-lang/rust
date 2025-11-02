// Regression test for the projection bug in <https://github.com/rust-lang/rust/issues/123953>
//
//@ compile-flags: -Zincremental-verify-ich=yes
//@ incremental

pub trait A {}
pub trait B: A {}

pub trait Mirror {
    type Assoc: ?Sized;
}

impl<T: ?Sized> Mirror for A {
    //~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates [E0207]
    //~| WARN trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    type Assoc = T;
}

pub fn foo<'a>(
    x: &'a <dyn A + 'static as Mirror>::Assoc
) -> &'a <dyn B + 'static as Mirror>::Assoc {
    //~^ ERROR the trait bound `(dyn B + 'static): Mirror` is not satisfied [E0277]
    //~| ERROR the trait bound `(dyn B + 'static): Mirror` is not satisfied [E0277]
    static
} //~ ERROR expected identifier, found `}`

pub fn main() {}
