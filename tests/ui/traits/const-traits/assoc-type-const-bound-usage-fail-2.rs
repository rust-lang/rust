//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// Check that `[const]` item bounds only hold if the where clauses on the
// associated type are also const.
// i.e. check that we validate the const conditions for the associated type
// when considering one of implied const bounds.

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    type Assoc<U>: [const] Trait
    where
        U: [const] Other;

    fn func();
}

#[const_trait]
trait Other {}

const fn fails<T: [const] Trait, U: Other>() {
    T::Assoc::<U>::func();
    //~^ ERROR the trait bound `U: [const] Other` is not satisfied
    <T as Trait>::Assoc::<U>::func();
    //~^ ERROR the trait bound `U: [const] Other` is not satisfied
}

const fn works<T: [const] Trait, U: [const] Other>() {
    T::Assoc::<U>::func();
    <T as Trait>::Assoc::<U>::func();
}

fn main() {}
