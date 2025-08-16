//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// See the code below.
//
// We were using `DeepRejectCtxt` to ensure that `assemble_inherent_candidates_from_param`
// did not rely on the param-env being eagerly normalized. Since aliases unify with all
// types, this meant that a rigid param-env candidate like `<T as Deref>::Target: Trait1`
// would be registered as a "WhereClauseCandidate", which is treated as inherent. Since
// we evaluate these candidates for all self types in the deref chain, this candidate
// would be satisfied for `<T as Deref>::Target`, meaning that it would be preferred over
// an "extension" candidate like `<T as Deref>::Target: Trait2` even though it holds.
// This is problematic, since it causes ambiguities to be broken somewhat arbitrarily.
// And as a side-effect, it also caused our computation of "used" traits to be miscalculated
// since inherent candidates don't count as an import usage.

use std::ops::Deref;

trait Trait1 {
    fn method(&self) {
        println!("1");
    }
}

trait Trait2 {
    fn method(&self) {
        println!("2");
    }
}
impl<T: Other + ?Sized> Trait2 for T {}

trait Other {}

fn foo<T>(x: T)
where
    T: Deref,
    <T as Deref>::Target: Trait1 + Other,
{
    // Make sure that we don't prefer methods from where clauses for rigid aliases,
    // just for params. We could revisit this behavior, but it would be a lang change.
    x.method();
    //~^ ERROR multiple applicable items in scope
}

fn main() {}
