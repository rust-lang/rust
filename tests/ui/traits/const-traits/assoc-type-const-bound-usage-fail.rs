//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// Check that `[const]` item bounds only hold if the parent trait is `[const]`.
// i.e. check that we validate the const conditions for the associated type
// when considering one of implied const bounds.

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    type Assoc: [const] Trait;
    fn func();
}

const fn unqualified<T: Trait>() {
    T::Assoc::func();
    //~^ ERROR the trait bound `T: [const] Trait` is not satisfied
    <T as Trait>::Assoc::func();
    //~^ ERROR the trait bound `T: [const] Trait` is not satisfied
}

const fn works<T: [const] Trait>() {
    T::Assoc::func();
    <T as Trait>::Assoc::func();
}

fn main() {}
