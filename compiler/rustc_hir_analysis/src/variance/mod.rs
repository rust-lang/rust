//! Module for inferring the variance of type and lifetime parameters. See the [rustc dev guide]
//! chapter for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/variance.html

use itertools::Itertools;
use rustc_arena::DroplessArena;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{
    self, CrateVariancesMap, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
};
use tracing::{debug, instrument};

/// Defines the `TermsContext` basically houses an arena where we can
/// allocate terms.
mod terms;

/// Code to gather up constraints.
mod constraints;

/// Code to solve constraints and write out the results.
mod solve;

pub(crate) mod dump;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { variances_of, crate_variances, ..*providers };
}

fn crate_variances(tcx: TyCtxt<'_>, (): ()) -> CrateVariancesMap<'_> {
    let arena = DroplessArena::default();
    let terms_cx = terms::determine_parameters_to_be_inferred(tcx, &arena);
    let constraints_cx = constraints::add_constraints_from_crate(terms_cx);
    solve::solve_constraints(constraints_cx)
}

fn variances_of(tcx: TyCtxt<'_>, item_def_id: LocalDefId) -> &[ty::Variance] {
    // Skip items with no generics - there's nothing to infer in them.
    if tcx.generics_of(item_def_id).is_empty() {
        return &[];
    }

    let kind = tcx.def_kind(item_def_id);
    match kind {
        DefKind::Fn
        | DefKind::AssocFn
        | DefKind::Enum
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Ctor(..) => {
            // These are inferred.
            let crate_map = tcx.crate_variances(());
            return crate_map.variances.get(&item_def_id.to_def_id()).copied().unwrap_or(&[]);
        }
        DefKind::TyAlias if tcx.type_alias_is_lazy(item_def_id) => {
            // These are inferred.
            let crate_map = tcx.crate_variances(());
            return crate_map.variances.get(&item_def_id.to_def_id()).copied().unwrap_or(&[]);
        }
        DefKind::AssocTy => match tcx.opt_rpitit_info(item_def_id.to_def_id()) {
            Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
                return variance_of_opaque(
                    tcx,
                    opaque_def_id.expect_local(),
                    ForceCaptureTraitArgs::Yes,
                );
            }
            None | Some(ty::ImplTraitInTraitData::Impl { .. }) => {}
        },
        DefKind::OpaqueTy => {
            let force_capture_trait_args = if let hir::OpaqueTyOrigin::FnReturn {
                parent: _,
                in_trait_or_impl: Some(hir::RpitContext::Trait),
            } =
                tcx.hir_node_by_def_id(item_def_id).expect_opaque_ty().origin
            {
                ForceCaptureTraitArgs::Yes
            } else {
                ForceCaptureTraitArgs::No
            };

            return variance_of_opaque(tcx, item_def_id, force_capture_trait_args);
        }
        _ => {}
    }

    // Variance not relevant.
    span_bug!(
        tcx.def_span(item_def_id),
        "asked to compute variance for {}",
        kind.descr(item_def_id.to_def_id())
    );
}

#[derive(Debug, Copy, Clone)]
enum ForceCaptureTraitArgs {
    Yes,
    No,
}

#[instrument(level = "trace", skip(tcx), ret)]
fn variance_of_opaque(
    tcx: TyCtxt<'_>,
    item_def_id: LocalDefId,
    force_capture_trait_args: ForceCaptureTraitArgs,
) -> &[ty::Variance] {
    let generics = tcx.generics_of(item_def_id);

    // Opaque types may only use regions that are bound. So for
    // ```rust
    // type Foo<'a, 'b, 'c> = impl Trait<'a> + 'b;
    // ```
    // we may not use `'c` in the hidden type.
    struct OpaqueTypeLifetimeCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        root_def_id: DefId,
        variances: Vec<ty::Variance>,
    }

    impl<'tcx> OpaqueTypeLifetimeCollector<'tcx> {
        #[instrument(level = "trace", skip(self), ret)]
        fn visit_opaque(&mut self, def_id: DefId, args: GenericArgsRef<'tcx>) {
            if def_id != self.root_def_id && self.tcx.is_descendant_of(def_id, self.root_def_id) {
                let child_variances = self.tcx.variances_of(def_id);
                for (a, v) in args.iter().zip_eq(child_variances) {
                    if *v != ty::Bivariant {
                        a.visit_with(self);
                    }
                }
            } else {
                args.visit_with(self)
            }
        }
    }

    impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeLifetimeCollector<'tcx> {
        #[instrument(level = "trace", skip(self), ret)]
        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            if let ty::RegionKind::ReEarlyParam(ebr) = r.kind() {
                self.variances[ebr.index as usize] = ty::Invariant;
            }
        }

        #[instrument(level = "trace", skip(self), ret)]
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            match t.kind() {
                ty::Alias(ty::Opaque, ty::AliasTy { def_id, args, .. }) => {
                    self.visit_opaque(*def_id, args);
                }
                _ => t.super_visit_with(self),
            }
        }
    }

    // By default, RPIT are invariant wrt type and const generics, but they are bivariant wrt
    // lifetime generics.
    let mut variances = vec![ty::Invariant; generics.count()];

    // Mark all lifetimes from parent generics as unused (Bivariant).
    // This will be overridden later if required.
    {
        let mut generics = generics;
        while let Some(def_id) = generics.parent {
            generics = tcx.generics_of(def_id);

            // Don't mark trait params generic if we're in an RPITIT.
            if matches!(force_capture_trait_args, ForceCaptureTraitArgs::Yes)
                && generics.parent.is_none()
            {
                debug_assert_eq!(tcx.def_kind(def_id), DefKind::Trait);
                break;
            }

            for param in &generics.own_params {
                match param.kind {
                    ty::GenericParamDefKind::Lifetime => {
                        variances[param.index as usize] = ty::Bivariant;
                    }
                    ty::GenericParamDefKind::Type { .. }
                    | ty::GenericParamDefKind::Const { .. } => {}
                }
            }
        }
    }

    let mut collector =
        OpaqueTypeLifetimeCollector { tcx, root_def_id: item_def_id.to_def_id(), variances };
    let id_args = ty::GenericArgs::identity_for_item(tcx, item_def_id);
    for (pred, _) in tcx.explicit_item_bounds(item_def_id).iter_instantiated_copied(tcx, id_args) {
        debug!(?pred);

        // We only ignore opaque type args if the opaque type is the outermost type.
        // The opaque type may be nested within itself via recursion in e.g.
        // type Foo<'a> = impl PartialEq<Foo<'a>>;
        // which thus mentions `'a` and should thus accept hidden types that borrow 'a
        // instead of requiring an additional `+ 'a`.
        match pred.kind().skip_binder() {
            ty::ClauseKind::Trait(ty::TraitPredicate {
                trait_ref: ty::TraitRef { def_id: _, args, .. },
                polarity: _,
            })
            | ty::ClauseKind::HostEffect(ty::HostEffectPredicate {
                trait_ref: ty::TraitRef { def_id: _, args, .. },
                constness: _,
            }) => {
                for arg in &args[1..] {
                    arg.visit_with(&mut collector);
                }
            }
            ty::ClauseKind::Projection(ty::ProjectionPredicate {
                projection_term: ty::AliasTerm { args, .. },
                term,
            }) => {
                for arg in &args[1..] {
                    arg.visit_with(&mut collector);
                }
                term.visit_with(&mut collector);
            }
            ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(_, region)) => {
                region.visit_with(&mut collector);
            }
            _ => {
                pred.visit_with(&mut collector);
            }
        }
    }
    tcx.arena.alloc_from_iter(collector.variances)
}
