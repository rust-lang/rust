// We currently fail to wfcheck generic arguments that correspond to unused LTA generic parameters
// since we generally normalize types before wfchecking them, so we accidentally "expand them away"
// before we can check them.
//
// (We do still check predicates that reference unused parameters, of course.)
//
// FIXME(lazy_type_alias): I consider #100041 to be a stabilization-blocking concern for the checked
//                         version of LTA! The *entire premise* of checked_LTA is wfchecking;
//                         we can't have such obvious holes in it!
//
//@ revisions: current-bugged next-bugged next-fixed
//
//@[next-bugged] compile-flags: -Znext-solver
//@[next-fixed] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//
//@[current-bugged] known-bug: #100041
//@[current-bugged] check-pass
//@[next-bugged] known-bug: #100041
//@[next-bugged] check-pass

#![feature(lazy_type_alias)]

type A<T: ?Sized> = ();

type A0 = A<[str]>; // FIXME: `str: Sized` unsatisfied
type A1<'r> = A<&'static &'r ()>; // FIXME: `'r: 'static` unsatisfied
#[cfg(not(next_bugged))] type A2 = A<[(); panic!()]>; // FIXME: `panic!()` diverging
//[next-fixed]~^ ERROR evaluation panicked

#[cfg(not(next_bugged))] const _: A<[str]> = (); // FIXME: `str: Sized` unsatisfied
//[next-fixed]~^ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
