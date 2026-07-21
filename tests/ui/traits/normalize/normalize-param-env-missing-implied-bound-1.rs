//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// In `fn hrtb` normalizing the elaborated where-clause
// `T: for<'a> Supertrait<<&'a () as WithAssoc>::As>` results in
// a region error as `&'a ()` is not equal to `&'static ()`.
//
// This happens even though the where-clauses themselves are well-formed
// as we never have to prove `&'a (): WithAssoc` as `'a` is a bound variable.
//
// I don't think this can cause unsoundness and has already been accepted in the
// old solver as we didn't register TypeOutlives constraints if `ignoring_regions`
// is set. The new solver always registers outlives constraints, so this then
// caused an error there. We need to ignore these region errors with the new solver
// due to trait-system-refactor-initiative#166.

#![allow(unused)]

trait Supertrait<T> {}

trait Other {
    fn method(&self) {}
}

impl<T: 'static> WithAssoc for T {
    type As = T;
}

trait WithAssoc {
    type As;
}

trait Trait<P: WithAssoc>: Supertrait<P::As> {
    fn method(&self) {}
}

fn hrtb<T: for<'a> Trait<&'a ()>>() {}
//~^ ERROR the type `&'a ()` does not fulfill the required lifetime

pub fn main() {}
