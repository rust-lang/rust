//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//
// This demonstrates that the `has_non_region_infer()`
// disjunctive check in the `TypeOutlives` handler in
// `evaluate_predicate_recursively` is load bearing with respect to
// marker trait winnowing.
//
// It's harder than one might imagine to demonstrate this.  The outer
// obligation must be a marker trait predicate, otherwise winnowing
// gives up on any nontrivial multi-candidate case.  This prevents us
// from simply making the outer obligation a type-outlives obligation.
//
// But the outer obligation must also be free of nonregion infer
// vars.  Otherwise, when we're considering candidates, each
// call to `prefer_lhs_over_victim` will return `false` and
// we'll get ambiguity.
//
// Together, these imply that the infer var must come from the
// impl.  It can't come from our starting obligation.  So we need a
// subobligation, produced by candidate evaluation, that is both 1) a
// `TypeOutlives` predicate and 2) has an infer var.
//
// To make the would-be bug visible, we need 1) the candidate with the
// infer var to be picked as the winner, 2) the winner's WCs to fail,
// and 3) at least one other candidate in the winnowing set that gets
// marked `EvaluatedToOK`.
//
// How do we introduce an inference variable from an impl?  All
// inference variables used in the trait ref (the trait and its
// generic arguments, including `Self`) will have been resolved
// and substituted by this point.  We can't simply constrain a type
// parameter only by a lifetime -- that will produce an error about
// the parameter not being constrained (E0207).  Fortunately, if
// we bind that parameter to a trait associated type it will be
// treated as constrained.  Then, we can add a type-outlives bound
// on it.  This then survives as an infer var all the way to the
// `TypeOutlives` handler.
//
// Without the `has_non_region_infer()` check in the `TypeOutlives`
// handler, this `?U: 'static` would produce `EvaluatedToOk` rather
// than `EvaluatedToOkModuloRegions`.  Because this impl appears
// second, marker trait winnowing would pick it as the winner and we'd
// register the `?U: 'static` obligation.  That would then fail, after
// we substitute `?U = X`, and we'd get a spurious "may not live long
// enough" (E0310) error.
//
// With the check, the second impl gets `EvaluatedToOkModuloRegions`,
// so the first impl correctly wins.
//
// See #153847.
#![feature(marker_trait_attr)]

trait Assoc {
    type Ty;
}
impl<T> Assoc for T {
    type Ty = T;
}

#[marker]
trait Marker {}

impl<T> Marker for T where T: Assoc {}
impl<T, U> Marker for T
where
    T: Assoc<Ty = U>,
    U: 'static,
{
}

fn requires<T: Marker>() {}
fn use_marker<X>() {
    requires::<X>();
}

fn main() {}
