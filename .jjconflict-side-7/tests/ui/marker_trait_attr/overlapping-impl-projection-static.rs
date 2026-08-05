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
// This test is interesting because the projection reaches the
// `TypeOutlives` handler unnormalized.
//
// If not for the `has_non_region_param()` disjunct, the
// `<T as Assoc>::Ty: 'static` would get `EvaluatedToOk` rather
// than `EvaluatedToOkModuloRegions` and cause winnowing to select
// the second impl.  The `'static` bound then gets registered as
// a region obligation that borrowck can't discharge, causing a
// spurious "may not live long enough" (E0310) error to be reported
// against the first impl.
//
// With the check, the second impl gets `EvaluatedToOkModuloRegions`,
// so the first impl correctly wins.
//
// See #153847.
#![feature(marker_trait_attr)]

#[marker]
trait Marker {}

trait Assoc {
    type Ty;
}
impl<T> Assoc for T {
    type Ty = T;
}

impl<T: Assoc> Marker for T where <T as Assoc>::Ty: Copy {}
impl<T: Assoc> Marker for T where <T as Assoc>::Ty: 'static {}

fn main() {}
