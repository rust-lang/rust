//@ check-pass

// Regression test for #157122.
//
// When WF-checking a type, the visitor recurses through higher-ranked binders
// without instantiating them, so a `dyn Trait` nested inside a `for<'a>` binder
// reaches the `ty::Dynamic` arm of `WfPredicates::visit_ty` while still carrying
// escaping bound vars. Building the principal trait ref there via
// `ExistentialTraitRef::with_self_ty` passed that escaping self type to a
// `debug_assert!(!self_ty.has_escaping_bound_vars())`, which ICEs once the
// assertion is enabled. Creating a trait ref with an escaping self type is fine
// -- escaping bound vars are caught where they are actually used -- so the
// assertion was removed rather than worked around. The `ConstArgHasType` check
// this arm reads off still runs; see `wf-dyn-in-hrtb-bound-const-mismatch.rs`.
// Distilled from `itertools`'s `FormatWith` `Display` impl.

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
