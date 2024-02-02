// compile-flags: -Znext-solver
//
// This is a gnarly test but I don't know how to minimize it, frankly.

#![feature(lazy_type_alias)]
//~^ WARN the feature `lazy_type_alias` is incomplete

trait ToUnit<'a> {
    type Unit;
}

trait Overlap<T> {}

type Assoc<'a, T> = <*const T as ToUnit<'a>>::Unit;
//~^ ERROR: not well-formed

impl<T> Overlap<T> for T {}

impl<T> Overlap<for<'a> fn(Assoc<'a, T>)> for T where Missing: Overlap<T> {}
//~^ ERROR conflicting implementations of trait `Overlap<fn(_)>` for type `fn(_)`
//~| ERROR cannot find type `Missing` in this scope

fn main() {}
