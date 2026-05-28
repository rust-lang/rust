// Check that we accept unused generic parameters on lazy type aliases (for context, we reject
// unused type parameters on eager type aliases).
//
// As long as we check well-formedness before normalization there shouldn't be anything wrong with
// such parameters since we know that the corresponding arguments will get wfchecked regardless.
//
// FIXME(lazy_type_alias, #100041): At the time of writing however, that's not the case. I consider
//                                  this to be stabilization-blocking concern for the strong /
//                                  checked version of LTA!
//                                  See also `unused-generic-arguments-not-wfchecked.rs`.
//
//                                  (We *do* ofc still detect unsatisfied predicates even if they
//                                  reference unused parameters)
//
// issue: <https://github.com/rust-lang/rust/issues/140230>
//
//@ revisions: pass fail
//@[pass] check-pass

#![feature(lazy_type_alias)]

type A<'a: 'static> = ();
const _: A<'static> = ();
#[cfg(fail)] fn a(_: A<'_>) {} //[fail]~ ERROR lifetime bound not satisfied

type B<T> = ();
const _: B<i32> = ();
#[cfg(fail)] fn b(_: B<str>) {}
//[fail]~^ ERROR the size for values of type `str` cannot be known at compilation time
//[fail]~| ERROR the size for values of type `str` cannot be known at compilation time

type C<const N: usize> = ();
const _: C<{ 0 * 1 }> = ();
#[cfg(fail)] fn c(_: C<{ panic!() }>) {} //[fail]~ ERROR evaluation panicked

fn main() {}
