//! Module for inferring the variance of type and lifetime parameters. See the [rustc dev guide]
//! chapter for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/variance.html

use rustc_arena::DroplessArena;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, CrateVariancesMap, TyCtxt};

/// Defines the `TermsContext` basically houses an arena where we can
/// allocate terms.
mod terms;

/// Code to gather up constraints.
mod constraints;

/// Code to solve constraints and write out the results.
mod solve;

/// Code to write unit tests of variance.
pub mod test;

/// Code for transforming variances.
mod xform;

pub fn provide(providers: &mut Providers) {
    *providers = Providers { variances_of, crate_variances, ..*providers };
}

fn crate_variances(tcx: TyCtxt<'_>, (): ()) -> CrateVariancesMap<'_> {
    let arena = DroplessArena::default();
    let terms_cx = terms::determine_parameters_to_be_inferred(tcx, &arena);
    let constraints_cx = constraints::add_constraints_from_crate(terms_cx);
    solve::solve_constraints(constraints_cx)
}

fn variances_of(tcx: TyCtxt<'_>, item_def_id: DefId) -> &[ty::Variance] {
    // Skip items with no generics - there's nothing to infer in them.
    if tcx.generics_of(item_def_id).count() == 0 {
        return &[];
    }

    match tcx.def_kind(item_def_id) {
        DefKind::Fn
        | DefKind::AssocFn
        | DefKind::Enum
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Variant
        | DefKind::Ctor(..) => {}
        _ => {
            // Variance not relevant.
            span_bug!(tcx.def_span(item_def_id), "asked to compute variance for wrong kind of item")
        }
    }

    // Everything else must be inferred.

    let crate_map = tcx.crate_variances(());
    crate_map.variances.get(&item_def_id).copied().unwrap_or(&[])
}
