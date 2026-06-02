//@ check-pass

// Regression test for #157122.
//
// When WF-checking a type, the visitor recurses through higher-ranked binders
// without instantiating them, so a `dyn Trait` nested inside a `for<'a>` binder
// reaches the `ty::Dynamic` arm of `WfPredicates::visit_ty` while still carrying
// escaping bound vars. Building the principal trait ref there via
// `ExistentialTraitRef::with_self_ty` fed that escaping self type straight into
// `debug_assert!(!self_ty.has_escaping_bound_vars())`, causing an ICE. We now
// substitute a placeholder self type for the escaping case (the self type is
// irrelevant to the `ConstArgHasType` clauses this code reads off), so the
// assertion holds and the const-argument check is still performed -- see
// `wf-dyn-in-hrtb-bound-const-mismatch.rs` for the latter. Distilled from
// `itertools`'s `FormatWith` `Display` impl.

use std::fmt;

fn call<F>(mut f: F)
where
    F: FnMut(&mut dyn FnMut(&dyn fmt::Display) -> fmt::Result) -> fmt::Result,
{
    let _ = f(&mut |_disp: &dyn fmt::Display| Ok(()));
}

fn main() {
    call(|cb| cb(&0i32));
}
