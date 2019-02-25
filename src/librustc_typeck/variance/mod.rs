//! Module for inferring the variance of type and lifetime parameters. See the [rustc guide]
//! chapter for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/variance.html

use arena;
use rustc::hir;
use hir::Node;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::ty::{self, CrateVariancesMap, TyCtxt};
use rustc::ty::query::Providers;
use rustc_data_structures::sync::Lrc;

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

pub fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        variances_of,
        crate_variances,
        ..*providers
    };
}

fn crate_variances<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, crate_num: CrateNum)
                             -> Lrc<CrateVariancesMap> {
    assert_eq!(crate_num, LOCAL_CRATE);
    let mut arena = arena::TypedArena::default();
    let terms_cx = terms::determine_parameters_to_be_inferred(tcx, &mut arena);
    let constraints_cx = constraints::add_constraints_from_crate(terms_cx);
    Lrc::new(solve::solve_constraints(constraints_cx))
}

fn variances_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, item_def_id: DefId)
                          -> Lrc<Vec<ty::Variance>> {
    let id = tcx.hir().as_local_hir_id(item_def_id).expect("expected local def-id");
    let unsupported = || {
        // Variance not relevant.
        span_bug!(tcx.hir().span_by_hir_id(id), "asked to compute variance for wrong kind of item")
    };
    match tcx.hir().get_by_hir_id(id) {
        Node::Item(item) => match item.node {
            hir::ItemKind::Enum(..) |
            hir::ItemKind::Struct(..) |
            hir::ItemKind::Union(..) |
            hir::ItemKind::Fn(..) => {}

            _ => unsupported()
        },

        Node::TraitItem(item) => match item.node {
            hir::TraitItemKind::Method(..) => {}

            _ => unsupported()
        },

        Node::ImplItem(item) => match item.node {
            hir::ImplItemKind::Method(..) => {}

            _ => unsupported()
        },

        Node::ForeignItem(item) => match item.node {
            hir::ForeignItemKind::Fn(..) => {}

            _ => unsupported()
        },

        Node::Variant(_) | Node::StructCtor(_) => {}

        _ => unsupported()
    }

    // Everything else must be inferred.

    let crate_map = tcx.crate_variances(LOCAL_CRATE);
    crate_map.variances.get(&item_def_id)
                       .unwrap_or(&crate_map.empty_variance)
                       .clone()
}
