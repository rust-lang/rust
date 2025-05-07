//@ compile-flags: -Znext-solver
//
// This is a gnarly test but I don't know how to minimize it, frankly.

#![feature(lazy_type_alias)]
//~^ WARN the feature `lazy_type_alias` is incomplete

trait ToUnit<'a> {
    type Unit;
}

trait Overlap<T> {}

type Assoc<'a, T> = <*const T as ToUnit<'a>>::Unit;
//~^ ERROR the trait bound `*const T: ToUnit<'a>` is not satisfied

impl<T> Overlap<T> for T {}

impl<T> Overlap<for<'a> fn(Assoc<'a, T>)> for T where Missing: Overlap<T> {}
//~^ ERROR cannot find type `Missing` in this scope
//~| ERROR the trait bound `T: Overlap<for<'a> fn(Assoc<'a, T>)>` is not satisfied
//~| ERROR the trait bound `for<'a> *const T: ToUnit<'a>` is not satisfied

fn main() {}
