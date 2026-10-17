//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//
// This demonstrates that the `has_non_region_param()`
// disjunctive check in the `TypeOutlives` handler in
// `evaluate_predicate_recursively` is load bearing with respect to
// marker trait winnowing.
//
// Conversely, this does *not* show that the `has_non_region_infer()`
// check is load bearing -- you could be forgiven for thinking that
// it would.  The `f::<u8, _>` will result in a `u8: Marker<?U>`
// obligation; candidate evaluation will produce a `?U: 'static`
// subobligation.  The infer var *will* reach the check in
// question.  But then, in `prefer_lhs_over_victim`, we check for
// `!has_non_region_infer` (there's a long comment there describing
// the subtle reason why).  This causes us to defer making a
// selection.  At a later point, when we check again, we'll have
// equated and substituted `?U = X`, so there won't be an infer var,
// but there will be a type parameter.
//
// Without the `has_non_region_param()` check in the
// `TypeOutlives` handler, the substituted `X: 'static` obligation
// from the impl would produce `EvaluatedToOk` rather than
// `EvaluatedToOkModuloRegions`.  Because this impl appears second,
// marker trait winnowing would pick it as the winner and we'd
// register the `X: 'static` obligation.  That would then fail, and
// we'd get a spurious "may not live long enough" (E0310) error.
//
// With the check, the second impl gets `EvaluatedToOkModuloRegions`,
// so the first impl correctly wins.
//
// See #153847.
#![feature(marker_trait_attr)]

#[marker]
trait Marker<T> {}

impl<T> Marker<T> for u8 {}
impl<T: 'static> Marker<T> for u8 {}

fn f<T: Marker<U>, U>() -> U {
    loop {}
}

fn g<X>() {
    let x = f::<u8, _>();
    let _: X = x;
}

fn main() {}
