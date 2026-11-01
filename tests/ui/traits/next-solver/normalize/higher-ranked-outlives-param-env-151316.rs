//! Higher-ranked outlives bounds on projections in where clauses used to ICE
//! with "could not normalize" in the new solver.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/151316>.
//@ edition: 2024
//@ compile-flags: -Znext-solver

#![feature(unboxed_closures)]
#![feature(non_lifetime_binders)]

struct S<T>(T)
//~^ ERROR expected `FnOnce(&'a mut i32)` signature, found `i32`
where
    for<F> F:,
    for<'a> <i32 as FnOnce<(&'a mut i32,)>>::Output: Future + 'a;

fn wrapper<F>()
//~^ ERROR expected `FnOnce<&'a mut i32>` signature, found `i32`
where
    F:,
    for<'a> <i32 as FnOnce<&'a mut i32>>::Output: Future + 'a,
{
}

trait Trait<'a, 'b> {
    type Assoc;
}

struct ImpliedBound<'a, 'c, T>(T, &'a (), &'c ())
//~^ ERROR the trait bound `for<'b> T: Trait<'a, 'b>` is not satisfied
where
    for<'b> <T as Trait<'a, 'b>>::Assoc: 'c;

fn main() {}
