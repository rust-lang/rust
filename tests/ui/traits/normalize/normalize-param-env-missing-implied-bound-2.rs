//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// In `fn hrtb` normalizing the elaborated where-clause
// `T: for<'a> Supertrait<<&'a () as WithAssoc>::As>` results in
// a region error as `&'a ()` is not `'static`.
//
// This happens even though the where-clauses themselves are well-formed
// as we never have to prove `&'a (): WithAssoc` as `'a` is a bound variable.
//
// Unlike `normalize-param-env-missing-implied-bound-1.rs` this snippet caused
// an ICE with the old trait solver, as we did register region constraints from
// relating types. This ICE was tracked in #136661.

#![allow(unused)]

trait Supertrait<T> {}

trait Other {
    fn method(&self) {}
}

impl WithAssoc for &'static () {
    type As = ();
}

trait WithAssoc {
    type As;
}

trait Trait<P: WithAssoc>: Supertrait<P::As> {
    fn method(&self) {}
}

fn hrtb<T: for<'a> Trait<&'a ()>>() {}

pub fn main() {}
