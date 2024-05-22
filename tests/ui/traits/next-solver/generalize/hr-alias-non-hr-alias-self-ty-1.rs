//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Generalizing an alias referencing escaping bound variables
// is hard. We previously didn't replace this alias with inference
// variables but did replace nested alias which do not reference
// any bound variables. This caused us to stall with the following
// goal, which cannot make any progress:
//
// <<T as Id>::Refl as HigherRanked>::Output<'a>
//    alias-relate
// <?unconstrained as HigherRanked>::Output<'a>
//
//
// cc trait-system-refactor-initiative#110

#![allow(unused)]
trait HigherRanked {
    type Output<'a>;
}
trait Id {
    type Refl: HigherRanked;
}

fn foo<T: Id>() -> for<'a> fn(<<T as Id>::Refl as HigherRanked>::Output<'a>) {
    todo!()
}

fn bar<T: Id>() {
    // we generalize here
    let x = foo::<T>();
}

fn main() {}
